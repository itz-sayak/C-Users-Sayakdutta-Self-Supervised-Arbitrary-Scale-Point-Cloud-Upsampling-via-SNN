import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from fd import config, datacore
from fd.trainer import Trainer
from fd.checkpoints import CheckpointIO
import gc

def check_memory(iteration):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"[GPU MEM] iter={iteration} allocated={allocated:.2f}GB reserved={reserved:.2f}GB"
    return ""

def validate_batch(batch):
    if batch is None:
        return False
    
    required_keys = ['input', 'len']
    for key in required_keys:
        if key not in batch:
            print(f"Warning: Batch missing key '{key}'")
            return False
    
    inputs = batch['input']
    targets = batch['len']
    
    if inputs.dim() not in [3, 4]:
        print(f"Warning: Invalid input dimensions: {inputs.dim()}")
        return False
    
    if torch.isnan(inputs).any() or torch.isnan(targets).any():
        print("Warning: NaN detected in batch")
        return False
    
    return True

if __name__ == '__main__':
    cfg = config.load_config('config/fd.yaml')
    # Determine GPU device: respect CUDA_VISIBLE_DEVICES if set (remapped to 0..N-1),
    # otherwise use the first id from config.hardware.gpu_ids.
    gpu_ids = cfg.get('hardware', {}).get('gpu_ids', [0])
    if torch.cuda.is_available():
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            dev = 0
            print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} detected; using cuda:{dev}")
        else:
            dev = int(gpu_ids[0]) if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) > 0 else 0
            print(f"Using configured GPU id: {dev}")
        device = torch.device(f'cuda:{dev}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    cfg['training']['learning_rate'] = float(cfg['training'].get('learning_rate', 1e-4))
    cfg['training']['weight_decay'] = float(cfg['training'].get('weight_decay', 0.0))
    cfg['training']['batch_size'] = int(cfg['training'].get('batch_size', 4))
    cfg['training']['step_size'] = int(cfg['training'].get('step_size', 10000))
    cfg['training']['gamma'] = float(cfg['training'].get('gamma', 0.5))
    
    cfg['training']['max_epochs'] = int(cfg['training'].get('max_epochs', 200))
    cfg['training']['max_iterations'] = int(cfg['training'].get('max_iterations', 120000))
    cfg['training']['grad_clip'] = float(cfg['training'].get('grad_clip', 0.2))
    cfg['training']['use_amp'] = bool(cfg['training'].get('use_amp', True))
    
    t0 = time.time()

    out_dir = 'out/fd'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logfile = open(os.path.join(out_dir, 'log.txt'), 'a')
    batch_size = cfg['training']['batch_size']
    
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {cfg['training']['max_epochs']}")
    print(f"Max iterations: {cfg['training']['max_iterations']}")

    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    num_workers = cfg['training'].get('num_workers', 0)
    pin_memory = True if device.type == 'cuda' else False
    worker_init = datacore.worker_init_fn if num_workers > 0 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=datacore.collate_remove_none,
        worker_init_fn=worker_init,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=datacore.collate_remove_none,
        worker_init_fn=worker_init,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    print("Initializing model...")
    model = config.get_model(cfg, device)
    
    nparameters = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {nparameters:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    logfile.write(f'Total number of parameters: {nparameters}\n')
    logfile.write(f'Trainable parameters: {trainable_params}\n')
    
    snn_params = 0
    for name, p in model.named_parameters():
        if 'membrane' in name or 'threshold' in name or 'refractory' in name:
            snn_params += p.numel()
    print(f"SNN-specific parameters: {snn_params:,}")
    logfile.write(f'SNN parameters: {snn_params}\n')

    try:
        optimizer = config.get_optimizer(cfg, model)
        print("Using config.get_optimizer()")
    except:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['training']['learning_rate'],
            weight_decay=cfg['training']['weight_decay']
        )
        print("Using fallback optimizer")
    
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {cfg['training']['learning_rate']}")
    print(f"Weight decay: {cfg['training']['weight_decay']}")

    scheduler = None
    scheduler_type = cfg['training'].get('scheduler', 'step')
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg['training']['step_size'],
            gamma=cfg['training']['gamma']
        )
        print(f"StepLR scheduler: step_size={cfg['training']['step_size']}, gamma={cfg['training']['gamma']}")
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg['training'].get('factor', 0.5),
            patience=cfg['training'].get('patience', 10)
        )
        print("ReduceLROnPlateau scheduler")

    trainer = Trainer(model, optimizer, device=device)
    print("Trainer initialized")

    use_amp = cfg['training']['use_amp'] and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision training (AMP) enabled")

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    try:
        load_dict = checkpoint_io.load('model.pt')
        print("Checkpoint loaded successfully")
    except FileNotFoundError:
        load_dict = dict()
        print("No checkpoint found, starting from scratch")
    except Exception as e:
        print(f"Warning: error loading checkpoint: {e}")
        load_dict = dict()

    epoch_it = load_dict.get('epoch_it', 0)
    it = load_dict.get('it', 0)
    metric_val_best = load_dict.get('loss_val_best', np.inf)
    
    print(f"Starting from epoch {epoch_it}, iteration {it}")
    print(f"Previous best validation loss: {metric_val_best:.6f}")

    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    max_epochs = cfg['training']['max_epochs']
    max_iterations = cfg['training']['max_iterations']
    grad_clip = cfg['training']['grad_clip']

    mem_check_every = max(1, (print_every * 10))
    
    train_losses = []
    batch_times = []
    start_time = time.time()

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    try:
        while epoch_it < max_epochs and it < max_iterations:
            epoch_it += 1
            logfile.flush()
            
            print(f"\n--- Epoch {epoch_it}/{max_epochs} ---")
            epoch_start_time = time.time()
            
            if hasattr(model, 'reset_states'):
                model.reset_states()
                print("Reset SNN states")
            
            model.train()
            epoch_losses = []
            
            for batch in train_loader:
                it += 1
                
                if not validate_batch(batch):
                    print(f"Skipping invalid batch at iteration {it}")
                    continue
                
                batch_start_time = time.time()
                
                try:
                    optimizer.zero_grad()
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            if hasattr(trainer, 'compute_loss_with_dict'):
                                loss, loss_dict = trainer.compute_loss_with_dict(batch)
                            else:
                                loss = trainer.compute_loss(batch)
                                loss_dict = {}
                        
                        scaler.scale(loss).backward()
                        
                        if grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if hasattr(trainer, 'compute_loss_with_dict'):
                            loss, loss_dict = trainer.compute_loss_with_dict(batch)
                        else:
                            loss = trainer.compute_loss(batch)
                            loss_dict = {}
                        
                        loss.backward()
                        
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        
                        optimizer.step()
                    
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if 'membrane_decay' in name:
                                param.data.clamp_(0.1, 0.99)
                            elif 'threshold_adapt' in name:
                                param.data.clamp_(0.001, 0.1)
                            elif 'refractory_decay' in name:
                                param.data.clamp_(0.1, 0.95)
                    
                    loss_value = loss.item()
                    epoch_losses.append(loss_value)
                    train_losses.append(loss_value)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM at iteration {it}. Skipping batch...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                logger.add_scalar('train/loss', loss_value, it)
                
                if loss_dict:
                    for key, value in loss_dict.items():
                        logger.add_scalar(f'train/{key}', value, it)
                
                current_lr = optimizer.param_groups[0]['lr']
                logger.add_scalar('train/learning_rate', current_lr, it)
                
                if torch.cuda.is_available() and (it % mem_check_every) == 0:
                    mem_info = check_memory(it)
                    print(mem_info)
                    logfile.write(mem_info + '\n')
                
                if torch.cuda.is_available() and (it % 100) == 0:
                    torch.cuda.empty_cache()
                
                if print_every > 0 and (it % print_every) == 0 and it > 0:
                    avg_loss = np.mean(train_losses[-print_every:]) if len(train_losses) >= print_every else np.mean(train_losses)
                    avg_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else np.mean(batch_times)
                    
                    msg = f'[Epoch {epoch_it:03d}] it={it:06d}, loss={loss_value:.6f} (avg: {avg_loss:.6f}), lr={current_lr:.2e}, time={avg_time:.3f}s'
                    logfile.write(msg + '\n')
                    print(msg)
                
                if (checkpoint_every > 0 and (it % checkpoint_every) == 0) and it > 0:
                    print(f'Saving checkpoint at iteration {it}')
                    logfile.write('Saving checkpoint\n')
                    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best)
                
                if validate_every > 0 and (it % validate_every) == 0 and it > 0:
                    print(f'Running validation at iteration {it}...')
                    
                    if hasattr(model, 'reset_states'):
                        model.reset_states()
                    
                    if hasattr(trainer, 'evaluate_with_metrics'):
                        metric_val, val_metrics = trainer.evaluate_with_metrics(val_loader)
                        
                        logger.add_scalar('val/loss', metric_val, it)
                        for key, value in val_metrics.items():
                            logger.add_scalar(f'val/{key}', value, it)
                        
                        print(f'Validation - Loss: {metric_val:.6f}, Metrics: {val_metrics}')
                        logfile.write(f'Validation loss: {metric_val:.6f}\n')
                        for key, value in val_metrics.items():
                            logfile.write(f'  {key}: {value:.6f}\n')
                    else:
                        metric_val = trainer.evaluate(val_loader)
                        metric_val = float(metric_val)
                        logger.add_scalar('val/loss', metric_val, it)
                        
                        print(f'Validation loss: {metric_val:.6f}')
                        logfile.write(f'Validation loss: {metric_val:.6f}\n')
                    
                    if metric_val < metric_val_best:
                        metric_val_best = metric_val
                        print(f'New best model! Loss: {metric_val_best:.6f}')
                        logfile.write(f'New best model (loss {metric_val_best:.6f})\n')
                        checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                           loss_val_best=metric_val_best)
                    
                    if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(metric_val)
                
                if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                
                if it >= max_iterations:
                    print(f"Reached maximum iterations: {max_iterations}")
                    break
            
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
            
            print(f'Epoch {epoch_it} completed in {epoch_time:.1f}s, avg loss: {avg_epoch_loss:.6f}')
            logfile.write(f'Epoch {epoch_it} completed, avg loss: {avg_epoch_loss:.6f}\n')
            
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if epoch_it >= max_epochs:
                print(f"Reached maximum epochs: {max_epochs}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"Total iterations: {it}")
        print(f"Best validation loss: {metric_val_best:.6f}")
        
        logfile.write(f"\nTraining completed\n")
        logfile.write(f"Total time: {total_time:.1f}s\n")
        logfile.write(f"Total iterations: {it}\n")
        logfile.write(f"Best validation loss: {metric_val_best:.6f}\n")
        
        checkpoint_io.save('model_final.pt', epoch_it=epoch_it, it=it,
                           loss_val_best=metric_val_best)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        checkpoint_io.save('model_interrupt.pt', epoch_it=epoch_it, it=it,
                           loss_val_best=metric_val_best)
        logfile.write("\nTraining interrupted by user\n")
    
    except Exception as e:
        print(f"\nUnexpected exception during training: {e}")
        logfile.write(f"\nException: {e}\n")
        
        try:
            print("Saving crash checkpoint...")
            checkpoint_io.save('model_crash.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        except Exception as se:
            print(f"Could not save crash checkpoint: {se}")
        
        raise
    
    finally:
        logfile.close()
        logger.close()
        print("\nTraining finished.")