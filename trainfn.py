import torch
import torch.optim as optim
import numpy as np
import os
import time
import warnings
from tensorboardX import SummaryWriter
from fn import config, datacore
from fn.trainer import Trainer
from fn.checkpoints import CheckpointIO
from torch.cuda.amp import GradScaler, autocast
import sys

warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':  
    cfg = config.load_config('config/fn.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    seed = cfg.get('hardware', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = cfg.get('hardware', {}).get('deterministic', False)
        torch.backends.cudnn.benchmark = cfg.get('hardware', {}).get('cudnn_benchmark', True)

    out_dir = 'out/fn'
    logfile_path = os.path.join(out_dir, 'log.txt')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    logfile = open(logfile_path, 'a')
    
    def log_message(msg):
        print(msg)
        logfile.write(msg + '\n')
        logfile.flush()
    
    batch_size = cfg['training']['batch_size']
    
    num_workers = cfg['training'].get('num_workers', 4)
    pin_memory = cfg['hardware'].get('pin_memory', True)

    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True,
        collate_fn=datacore.collate_remove_none,
        worker_init_fn=datacore.worker_init_fn,
        pin_memory=pin_memory,
        prefetch_factor=cfg['training'].get('prefetch_factor', 2) if num_workers > 0 else None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        collate_fn=datacore.collate_remove_none,
        worker_init_fn=datacore.worker_init_fn,
        pin_memory=pin_memory
    )

    model = config.get_model(cfg, device)
    
    optimizer_type = cfg['training'].get('optimizer', 'adam')
    lr = cfg['training'].get('lr', 1e-4)
    weight_decay = cfg['training'].get('weight_decay', 0.0)
    betas = tuple(cfg['training'].get('betas', [0.9, 0.999]))
    
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    
    gradient_accumulation = cfg['training'].get('gradient_accumulation', 1)
    
    use_amp = cfg['training'].get('use_amp', False)
    if use_amp:
        scaler = GradScaler()
        log_message(f"Using mixed precision training (AMP)")
    else:
        scaler = None
    
    grad_clip = cfg['training'].get('grad_clip', None)
    grad_clip_type = cfg['training'].get('grad_clip_type', 'norm')
    
    if cfg['training'].get('gradient_checkpointing', False):
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            log_message("Enabled gradient checkpointing")
    
    trainer = Trainer(
        model, 
        optimizer, 
        device=device,
        gradient_accumulation=gradient_accumulation,
        use_amp=use_amp,
        scaler=scaler,
        grad_clip=grad_clip,
        grad_clip_type=grad_clip_type
    )

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    
    epoch_it = -1
    it = -1
    metric_val_best = np.inf
    
    if cfg['checkpoint'].get('resume', False):
        resume_file = cfg['checkpoint'].get('resume_file', 'model.pt')
        try:
            load_dict = checkpoint_io.load(resume_file)
            epoch_it = load_dict.get('epoch_it', -1)
            it = load_dict.get('it', -1)
            metric_val_best = load_dict.get('loss_val_best', np.inf)
            log_message(f"Resumed from checkpoint: epoch={epoch_it}, iteration={it}, best_val_loss={metric_val_best:.6f}")
        except FileExistsError:
            log_message("No checkpoint found, starting from scratch")
    else:
        log_message("Starting training from scratch")

    if cfg['checkpoint'].get('pretrained', False):
        pretrained_file = cfg['checkpoint'].get('pretrained_file', '')
        if pretrained_file and os.path.exists(pretrained_file):
            try:
                checkpoint_io.load(pretrained_file)
                log_message(f"Loaded pretrained weights from {pretrained_file}")
            except Exception as e:
                log_message(f"Error loading pretrained weights: {e}")
    
    lr_policy = cfg['training'].get('lr_policy', 'none')
    if lr_policy == 'step':
        lr_decay = cfg['training'].get('lr_decay', 0.95)
        lr_decay_step = cfg['training'].get('lr_decay_step', 2000)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay)
        log_message(f"Using StepLR scheduler: step_size={lr_decay_step}, gamma={lr_decay}")
    elif lr_policy == 'cosine':
        max_iterations = cfg['training'].get('max_iterations', 300000)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)
        log_message(f"Using CosineAnnealingLR scheduler: T_max={max_iterations}")
    elif lr_policy == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5000, factor=0.5)
        log_message("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = None
        log_message("No learning rate scheduler")

    log_dir = os.path.join(out_dir, 'logs')
    logger = SummaryWriter(log_dir)
    log_message(f"TensorBoard logs will be saved to: {log_dir}")

    nparameters = sum(p.numel() for p in model.parameters())
    log_message(f'Total number of parameters: {nparameters:,}')
    
    log_message(f"Model architecture: {cfg['model']['type']}")
    log_message(f"  - k_values: {cfg['model']['k_values']}")
    log_message(f"  - emb_dims: {cfg['model']['emb_dims']}")
    log_message(f"  - time_steps: {cfg['model']['time_steps_enc']}/{cfg['model']['time_steps_dec']}")
    log_message(f"  - num_heads: {cfg['model']['num_heads']}")

    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training'].get('visualize_every', 0)
    backup_every = cfg['training'].get('backup_every', 50000)
    max_iterations = cfg['training'].get('max_iterations', 300000)
    max_epochs = cfg['training'].get('max_epochs', 500)
    warmup_steps = cfg['training'].get('warmup_steps', 0)
    warmup_factor = cfg['training'].get('warmup_factor', 0.1)
    
    early_stopping = cfg['training'].get('early_stopping', False)
    patience = cfg['training'].get('patience', 20000)
    min_delta = cfg['training'].get('min_delta', 0.0001)
    no_improve_count = 0
    
    snn_training = cfg['training'].get('snn_training', {})
    spike_rate_target = snn_training.get('spike_rate_target', 0.1)
    state_reset_freq = snn_training.get('state_reset_freq', 100)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log_message(f"Starting training...")
    log_message(f"  - Batch size: {batch_size}")
    log_message(f"  - Gradient accumulation: {gradient_accumulation}")
    log_message(f"  - Effective batch size: {batch_size * gradient_accumulation}")
    log_message(f"  - Max iterations: {max_iterations}")
    log_message(f"  - Max epochs: {max_epochs}")
    log_message(f"  - Early stopping patience: {patience}")

    start_time = time.time()
    
    while True:
        epoch_it += 1
        
        if it >= max_iterations:
            log_message(f"Reached maximum iterations ({max_iterations}), stopping.")
            break
        
        if epoch_it >= max_epochs:
            log_message(f"Reached maximum epochs ({max_epochs}), stopping.")
            break
        
        if hasattr(model, 'reset_states'):
            model.reset_states()
            if epoch_it > 0 and epoch_it % 10 == 0:
                log_message("Reset SNN states")
        
        if hasattr(model, 'get_spike_statistics'):
            spike_stats = model.get_spike_statistics()
            for key, value in spike_stats.items():
                if value > 0:
                    logger.add_scalar(f'snn/{key}', value, it)
        
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            it += 1
            
            if 'input' not in batch or batch['input'].shape[0] == 1:
                continue
            
            if state_reset_freq > 0 and it % state_reset_freq == 0:
                if hasattr(model, 'reset_states'):
                    model.reset_states()
            
            if warmup_steps > 0 and it < warmup_steps:
                warmup_factor_current = warmup_factor + (1 - warmup_factor) * (it / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * warmup_factor_current
            
            try:
                result = trainer.train_step(batch)
                
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        loss, loss_dict = result[0], result[1]
                    else:
                        loss, loss_dict = result[0], {}
                else:
                    loss, loss_dict = result, {}
                
                if isinstance(loss, torch.Tensor):
                    loss_value = loss.item()
                else:
                    loss_value = float(loss)
                
                logger.add_scalar('train/loss', loss_value, it)
                
                if loss_dict:
                    for key, value in loss_dict.items():
                        if isinstance(value, (int, float)):
                            logger.add_scalar(f'train/{key}', value, it)
                        elif isinstance(value, torch.Tensor):
                            logger.add_scalar(f'train/{key}', value.item(), it)
                
                epoch_losses.append(loss_value)
                
                current_lr = optimizer.param_groups[0]['lr']
                logger.add_scalar('train/lr', current_lr, it)
                
                if grad_clip and it % 100 == 0:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    logger.add_scalar('train/grad_norm', total_norm, it)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_message(f"WARNING: OOM at iteration {it}. Skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            if print_every > 0 and (it % print_every) == 0:
                avg_loss = np.mean(epoch_losses[-print_every:]) if epoch_losses else 0
                elapsed = time.time() - start_time
                samples_per_sec = (it * batch_size) / elapsed
                log_message(f'[Epoch {epoch_it:03d}] it={it:06d}, loss={loss_value:.6f}, avg_loss={avg_loss:.6f}, lr={current_lr:.6f}, samples/s={samples_per_sec:.1f}')
            
            if checkpoint_every > 0 and (it % checkpoint_every) == 0 and it > 0:
                checkpoint_path = f'model_{it:06d}.pt'
                log_message(f'Saving checkpoint: {checkpoint_path}')
                checkpoint_io.save(
                    checkpoint_path, 
                    epoch_it=epoch_it, 
                    it=it,
                    loss_val_best=metric_val_best,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None
                )
                
                checkpoint_io.save('model_latest.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
            
            if backup_every > 0 and (it % backup_every) == 0 and it > 0:
                backup_path = f'backup_model_{it:06d}.pt'
                log_message(f'Saving backup checkpoint: {backup_path}')
                checkpoint_io.save(backup_path, epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
            
            if validate_every > 0 and (it % validate_every) == 0 and it > 0:
                log_message(f"Running validation at iteration {it}...")
                
                if hasattr(model, 'reset_states'):
                    model.reset_states()
                
                try:
                    result = trainer.evaluate(val_loader)
                    
                    if isinstance(result, tuple):
                        if len(result) >= 3:
                            val_loss, val_confidence, val_metrics = result[0], result[1], result[2]
                        elif len(result) >= 2:
                            val_loss, val_confidence, val_metrics = result[0], result[1], {}
                        else:
                            val_loss, val_confidence, val_metrics = result[0], 0.0, {}
                    else:
                        val_loss, val_confidence, val_metrics = result, 0.0, {}
                    
                    if isinstance(val_loss, torch.Tensor):
                        val_loss_value = val_loss.item()
                    else:
                        val_loss_value = float(val_loss)
                    
                    if isinstance(val_confidence, torch.Tensor):
                        val_confidence_value = val_confidence.item()
                    else:
                        val_confidence_value = float(val_confidence)
                    
                    logger.add_scalar('val/loss', val_loss_value, it)
                    logger.add_scalar('val/confidence', val_confidence_value, it)
                    
                    if val_metrics:
                        for key, value in val_metrics.items():
                            if isinstance(value, (int, float)):
                                logger.add_scalar(f'val/{key}', value, it)
                            elif isinstance(value, torch.Tensor):
                                logger.add_scalar(f'val/{key}', value.item(), it)
                    
                    log_message(f'Validation at it={it}: loss={val_loss_value:.6f}, confidence={val_confidence_value:.6f}')
                    
                    if scheduler is not None:
                        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss_value)
                        else:
                            scheduler.step()
                    
                    if val_loss_value < metric_val_best - min_delta:
                        metric_val_best = val_loss_value
                        no_improve_count = 0
                        log_message(f'New best model (loss {metric_val_best:.6f})')
                        checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                           loss_val_best=metric_val_best,
                                           val_loss=val_loss_value,
                                           val_confidence=val_confidence_value)
                    else:
                        no_improve_count += validate_every
                        log_message(f'No improvement for {no_improve_count} iterations')
                        
                        if early_stopping and no_improve_count >= patience:
                            log_message(f'Early stopping triggered after {it} iterations')
                            break
                    
                except Exception as e:
                    log_message(f"Error during validation: {e}")
                    continue
            
            if visualize_every > 0 and (it % visualize_every) == 0 and it > 0:
                try:
                    pass
                except Exception as e:
                    log_message(f"Error during visualization: {e}")
        
        if early_stopping and no_improve_count >= patience:
            break
        
        if epoch_losses:
            epoch_avg_loss = np.mean(epoch_losses)
            logger.add_scalar('train/epoch_loss', epoch_avg_loss, epoch_it)
            log_message(f'Epoch {epoch_it} finished. Average loss: {epoch_avg_loss:.6f}')
        
        if torch.cuda.is_available() and epoch_it % 5 == 0:
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    log_message(f"Training completed in {total_time/3600:.2f} hours")
    log_message(f"Final best validation loss: {metric_val_best:.6f}")
    
    checkpoint_io.save('model_final.pt', epoch_it=epoch_it, it=it,
                       loss_val_best=metric_val_best,
                       training_time=total_time)
    
    logfile.close()
    logger.close()
    
    print(f"Training finished. Log saved to {logfile_path}")