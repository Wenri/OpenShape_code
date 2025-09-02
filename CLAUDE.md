# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenShape is a 3D shape representation model that learns joint embeddings between 3D point clouds, text, and images. The model is trained on multiple datasets (Objaverse, ShapeNet, ABO, 3D-FUTURE) and achieves state-of-the-art performance on zero-shot 3D classification tasks.

## Development Commands

### Training
```bash
# Basic training with default config
python3 src/main.py --trial_name experiment_name

# Training with custom batch size
python3 src/main.py dataset.train_batch_size=20 --trial_name bs_20

# Training with different model architectures
python3 src/main.py --trial_name pointbert_all model.name=PointBERT model.scaling=4 model.use_dense=True training.lr=0.0005 training.lr_decay_rate=0.967

# Training with different datasets
python3 src/main.py --trial_name no_lvis dataset.train_split=meta_data/split/train_no_lvis.json
python3 src/main.py --trial_name shapenet_only dataset.train_split=meta_data/split/ablation/train_shapenet_only.json
```

### Inference
```bash
# Run example inference with pre-trained model
python3 src/example.py
```

### Data Download
```bash
# Download and extract training/evaluation data (~205GB)
python3 download_data.py
```

### Authentication
```bash
# Login to Weights & Biases for training tracking
wandb login {YOUR_WANDB_ID}
```

## Code Architecture

### Core Components

- **`src/main.py`**: Main training entry point with distributed training support
- **`src/train.py`**: Training loop implementation with evaluation on ModelNet40, Objaverse-LVIS, and ScanObjectNN
- **`src/example.py`**: Inference example showing how to load models and extract features
- **`src/models/`**: Model implementations including:
  - `Minkowski.py`: SparseConv backbone using MinkowskiEngine
  - `pointnet*.py`: PointNet variants
  - `dgcnn.py`: Dynamic Graph CNN
  - `pointmlp.py`: PointMLP architecture
  - `ppat.py`: Point Patch Attention Transformer (PointBERT)
- **`src/data/`**: Data loading and preprocessing utilities
- **`src/configs/`**: YAML configuration files for different training setups

### Model Architectures

The codebase supports multiple 3D backbone architectures:
- **SparseConv**: Sparse convolution networks using MinkowskiEngine
- **PointBERT**: Transformer-based architecture with point patch attention
- **PointNet/PointNet++**: Classic point-based networks
- **DGCNN**: Dynamic graph convolution
- **PointMLP**: MLP-based architecture

### Training Configuration

Configuration is handled through YAML files in `src/configs/` with command-line overrides:
- Main config: `src/configs/train.yaml`
- Model scaling, learning rates, and data paths are configurable
- Uses OpenCLIP for text/image feature extraction

### Data Pipeline

- Point clouds are normalized and optionally augmented (rotation, color dropping)
- Training uses contrastive learning between 3D shapes and text/image embeddings
- Supports multiple text sources: original metadata, BLIP captions, retrieval text
- Hard negative mining using precomputed KNN indices

### Distributed Training

- Multi-GPU training supported via DistributedDataParallel
- Automatic experiment directory creation with timestamped runs
- Checkpointing includes model state, optimizer, scheduler, and best metrics
- Integration with Weights & Biases for experiment tracking

### Key Dependencies

- PyTorch for deep learning
- MinkowskiEngine for sparse convolution
- OpenCLIP for CLIP model variants
- Open3D for point cloud processing
- Weights & Biases for experiment tracking
- OmegaConf for configuration management

## Data Requirements

- Point clouds expected to be normalized to [-1,1] range
- Gravity axis alignment important (configurable via `y_up` parameter)
- Default: 10,000 points per shape with XYZ+RGB features (6D)
- Training data includes precomputed text and image features using OpenCLIP

## Model Checkpoints

Pre-trained models available on HuggingFace Hub:
- `OpenShape/openshape-pointbert-vitg14-rgb`: Best performing PointBERT model
- `OpenShape/openshape-spconv-all`: SparseConv model trained on all datasets
- Various ablation models (no-LVIS, ShapeNet-only, etc.)