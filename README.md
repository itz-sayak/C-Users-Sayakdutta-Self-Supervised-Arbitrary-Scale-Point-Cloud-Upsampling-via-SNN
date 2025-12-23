# Self-Supervised Arbitrary-Scale Point Cloud Upsampling via Spiking Neural Networks

A PyTorch implementation of point cloud upsampling using bio-inspired Spiking Neural Networks (SNNs) for feature extraction.

## 🎯 Project Objective

This project performs **arbitrary-scale point cloud upsampling** by:
1. **Predicting surface normals** at query points using an SNN-based encoder
2. **Estimating distances** from query points to the true surface
3. **Refining point positions** by moving along predicted normals by predicted distances

The key innovation is using **Spiking Neural Networks (SNNs) in the encoder** for temporal feature extraction, which provides:
- Bio-inspired temporal dynamics for feature learning
- Learnable membrane potentials and threshold adaptation
- Multi-scale graph convolutions with spike-based attention

---

## 📁 Project Structure

```
Fimproved/
├── config/
│   ├── fn.yaml              # Normal estimation model config
│   └── fd.yaml              # Distance estimation model config
├── fn/                      # Normal estimation module
│   ├── snn_coder.py         # SNN encoder + Standard decoder
│   ├── trainer.py           # Training loop
│   ├── datacore.py          # Dataset loader
│   ├── field.py             # Data field definitions
│   ├── transform.py         # Data augmentation
│   ├── config.py            # Config utilities
│   └── checkpoints.py       # Checkpoint I/O
├── fd/                      # Distance estimation module
│   ├── snn_coder.py         # SNN encoder + Standard decoder
│   ├── trainer.py           # Training loop
│   ├── datacore.py          # Dataset loader
│   ├── field.py             # Data field definitions
│   ├── transform.py         # Data augmentation
│   ├── config.py            # Config utilities
│   └── checkpoints.py       # Checkpoint I/O
├── data/
│   └── ShapeNet/            # Training dataset
│       ├── 02691156/        # Airplane
│       ├── 02828884/        # Bench
│       ├── 03001627/        # Chair
│       ├── 03211117/        # Display
│       ├── 04256520/        # Sofa
│       └── 04401088/        # Telephone
├── out/
│   ├── fn/                  # Normal model checkpoints
│   └── fd/                  # Distance model checkpoints
├── test/                    # Test input point clouds
├── testout/                 # Generated output point clouds
├── trainfn.py               # Train normal estimation model
├── trainfd.py               # Train distance estimation model
├── generate.py              # Inference script
├── generation.py            # Upsampling pipeline
├── dense.cpp                # Seed point generation (C++)
└── dense                    # Compiled dense binary
```

---

## 🏗️ Architecture

### Overall Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Point   │     │   Seed Point    │     │    Upsampled    │
│     Cloud       │ ──► │   Generation    │ ──► │   Point Cloud   │
│    (sparse)     │     │    (dense)      │     │    (dense)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  For each seed pt:  │
                    │  1. Find K neighbors│
                    │  2. Predict normal  │
                    │  3. Predict distance│
                    │  4. Move along n×d  │
                    └─────────────────────┘
```

### Model Architecture (SNN Encoder + Standard Decoder)

Both `fn` (normal estimation) and `fd` (distance estimation) models share a similar architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SNN ENCODER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [B, N, M, 3] (patches with K neighbors)                 │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │   Conv1D + LIF  │  Initial feature extraction                │
│  │   Neurons       │  with membrane decay & threshold adapt     │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │ Multi-Head SNN  │  Graph-based attention with:               │
│  │ Transformer ×3  │  - KNN graph construction                  │
│  │                 │  - Spiking Q, K, V projections             │
│  │                 │  - Position-aware attention                │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │  Global Pool +  │  Aggregate temporal spike features         │
│  │  LIF Neuron     │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│  Output: [B, emb_dims] (2048-dim feature vector)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STANDARD DECODER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │  Linear + BN +  │  Residual MLP blocks                       │
│  │  GELU + Skip    │  (256 → 128 → 64)                          │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │  Self-Attention │  Standard (non-spiking) attention          │
│  │  (optional)     │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │  Output Head    │  fn: [B, 3] normals                        │
│  │                 │  fd: [B, 1] distances                      │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SNN Components

#### Multi-Time Constant LIF Neuron
```python
# Learnable parameters per layer:
- membrane_decay:    τ_m ∈ [0.1, 0.99]  # Membrane time constant
- threshold_adapt:   η_θ ∈ [0.001, 0.1] # Threshold adaptation rate
- refractory_decay:  τ_r ∈ [0.1, 0.95]  # Refractory period decay
- threshold_base:    θ_0                 # Base firing threshold

# Forward dynamics:
membrane = membrane × τ_m × (1 - refractory) + input
spikes = surrogate_gradient(membrane - threshold)
membrane = membrane × (1 - spikes)  # Reset after spike
threshold = θ_0 + (threshold - θ_0) × 0.95 + η_θ × spikes
```

#### SNN State Management
- States are **reset at the start of each epoch** to prevent temporal leakage
- States are **detached** between batches to allow proper gradient flow
- Time steps are configurable (default: 4 for encoder)

---

## 📊 Training Data

### Dataset: ShapeNet

The model is trained on 6 ShapeNet categories:

| Category ID | Name      | Train | Val | Test |
|-------------|-----------|-------|-----|------|
| 02691156    | Airplane  | 516   | 71  | 58   |
| 02828884    | Bench     | 48    | 6   | 6    |
| 03001627    | Chair     | 164   | 22  | 18   |
| 03211117    | Display   | 36    | 5   | 4    |
| 04256520    | Sofa      | 23    | 3   | 3    |
| 04401088    | Telephone | 62    | 8   | 7    |
| **Total**   |           | **850** | **114** | **96** |

### Data Format

Each model folder contains:
```
data/ShapeNet/<category>/<model_id>/
├── pointcloud.npz    # Input point cloud (points: [N, 3])
├── fn.npz            # Ground truth normals (input: [N, 3], normal: [N, 3])
└── fd.npz            # Ground truth distances (input: [N, 3], len: [N, 1])
```

### Data Pipeline

```
Raw Points → Subsample (1024/2048) → Extract Patches → KNN Neighbors → Normalize → Model
                                           │
                                           ▼
                                    [N, M, 3] patches
                                    N = 8-16 patches
                                    M = 64-100 neighbors
```

---

## 🚀 Usage

### Prerequisites

```bash
# Create conda environment
conda create -n deepfill python=3.10
conda activate deepfill

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboardX numpy scikit-learn trimesh tqdm pyyaml
```

### Training

#### 1. Train Normal Estimation Model (fn)

```bash
python trainfn.py
```

**Key hyperparameters** (in `config/fn.yaml`):
- `batch_size`: 2-4 (depending on GPU memory)
- `emb_dims`: 512 (feature dimension)
- `k_values`: [12, 8, 6] (KNN neighbors at each scale)
- `time_steps_enc`: 4 (SNN time steps)
- `lr`: 0.0002

#### 2. Train Distance Estimation Model (fd)

```bash
python trainfd.py
```

**Key hyperparameters** (in `config/fd.yaml`):
- `batch_size`: 4
- `emb_dims`: 512
- `k`: 20 (KNN neighbors)
- `time_steps_enc`: 5
- `lr`: 0.0001

### Training Outputs

```
out/
├── fn/
│   ├── model_best.pt      # Best validation loss checkpoint
│   ├── model_latest.pt    # Latest checkpoint
│   ├── model_XXXXX.pt     # Periodic checkpoints
│   └── logs/              # TensorBoard logs
└── fd/
    ├── model_best.pt
    ├── model_latest.pt
    └── logs/
```

### Resume Training

Set in config:
```yaml
checkpoint:
  resume: true
  resume_file: 'model_best.pt'
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir out/fn/logs

# Watch GPU usage
watch -n 1 nvidia-smi

# Check training log
tail -f train_fn.log
```

---

## 🔮 Inference (Point Cloud Upsampling)

### Single File

```bash
python generate.py --input test/cow.xyz --output testout/cow_upsampled.xyz --target_points 8192
```

### Batch Processing

```bash
python generate.py --input_dir test/ --output_dir testout/ --ratio 4
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Input .xyz file | - |
| `--output` | Output .xyz file | - |
| `--input_dir` | Directory with input files | `test/` |
| `--output_dir` | Directory for outputs | `testout/` |
| `--target_points` | Target output points | 8192 |
| `--ratio` | Upsampling ratio | 4 |
| `--fn_checkpoint` | Normal model checkpoint | `out/fn/model_best.pt` |
| `--fd_checkpoint` | Distance model checkpoint | `out/fd/model_best.pt` |

### Upsampling Pipeline

```
1. Load sparse point cloud (.xyz)
2. Normalize to unit bounding box
3. Generate dense seed points (using dense.cpp)
4. For each seed point:
   a. Find K nearest neighbors from input
   b. Create local patch [K, 3]
   c. Predict normal using fn model → [3]
   d. Rotate patch to align normal with x-axis
   e. Predict distance using fd model → [1]
   f. Move seed point: new_pos = seed + normal × distance
5. Remove outliers (statistical filtering)
6. Denormalize to original scale
7. Save output (.xyz or .ply)
```

---

## 📈 Results

### Training Metrics

| Model | Final Loss | Metric |
|-------|------------|--------|
| fn (normal) | ~1.5 | Angular loss (radians) |
| fd (distance) | ~0.0015 | MSE loss |

### Qualitative Results

The model can upsample point clouds by 4-16× while preserving:
- Sharp edges and corners
- Surface continuity
- Fine geometric details

---

## 🔧 Configuration

### Reduce Memory Usage (OOM Issues)

If you encounter OOM errors, modify configs:

```yaml
# config/fn.yaml or config/fd.yaml
model:
  k_values: [8, 6, 4]        # Reduce from [12, 8, 6]
  emb_dims: 256              # Reduce from 512
  time_steps_enc: 3          # Reduce from 4
  d_model: 64                # Reduce from 128
  decoder_hidden_dims: [128, 64, 32]  # Reduce from [256, 128, 64]

training:
  batch_size: 2              # Reduce from 4
```

### GPU Selection

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python trainfn.py

# Set in config
hardware:
  gpu_ids: [0]
```

---

## 📚 References

- Based on: "Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation"
- SNN design inspired by: Spiking Neural Networks literature
- Point cloud processing: DGCNN, PointNet++

---

## 📝 License

This project is for research purposes only.

---

## 🙏 Acknowledgments

- ShapeNet dataset
- PyTorch team
- Original SAPCU authors

## **Metrics & Evaluation**

- **Combined metrics file:** `out/metrics/metrics_all_combined.json` — per-sample merged metrics (Chamfer, Hausdorff, normal errors, F-score, and multiple Sinkhorn variants).
- **Individual metric files:** Located under `out/metrics/` (examples: `metrics_testout_full.json`, `metrics_testout_fscore.json`, `metrics_testout_sinkhorn.json`, `metrics_testout_sinkhorn_tight.json`, `metrics_testout_sinkhorn_down4096.json`).

**Recompute Sinkhorn (GPU preferred):** The repository includes `scripts/compute_sinkhorn.py`. It will try `geomloss` first and fall back to a stable log-domain Torch implementation when geomloss fails.

Example (fast, default):
```bash
python3 scripts/compute_sinkhorn.py \
  --pred_dir testout \
  --gt_root data/ShapeNet_GT/gt \
  --out_json out/metrics/metrics_testout_sinkhorn.json \
  --device cuda --blur 0.05
```

Example (tighter accuracy, slower, may downsample GT to save memory):
```bash
python3 scripts/compute_sinkhorn.py \
  --pred_dir testout \
  --gt_root data/ShapeNet_GT/gt \
  --out_json out/metrics/metrics_testout_sinkhorn_tighter.json \
  --device cuda --blur 0.0005 --iters 3000 --double --downsample_gt 4096
```

**Merge all metrics into a single JSON:**
```bash
python3 scripts/merge_metrics.py
# writes: out/metrics/metrics_all_combined.json
```

**Notes:**
- If `geomloss` installs but raises internal errors on this system, the script automatically uses a numerically-stable log-domain Sinkhorn fallback.
- Tighter `--blur` and larger `--iters` improve approximation to true EMD but increase runtime and memory. Use `--downsample_gt` to reduce GT size when needed.



