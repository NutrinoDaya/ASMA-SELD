# ASMA-SELD: Advanced Sound Localization and Detection

**Professional implementation of Advanced Spatial-aware Multi-modal Attention for Sound Event Localization and Detection (ASMA-SELD)**

[![GitHub](https://img.shields.io/badge/GitHub-NutrinoDaya/ASMA--SELD-blue?logo=github)](https://github.com/NutrinoDaya/ASMA-SELD)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/NutrinoDaya/ASMA-SELD)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-green)](https://zenodo.org/records/17567932)

This repository contains the production-ready implementation of our ASMA-SELD model, developed for the DCASE 2025 Challenge Task 3. The system achieves state-of-the-art performance in sound event localization and detection using multimodal audio-visual fusion.

**🔗 Links:**
- **GitHub Repository**: [https://github.com/NutrinoDaya/ASMA-SELD](https://github.com/NutrinoDaya/ASMA-SELD)
- **Pre-trained Model**: [https://huggingface.co/NutrinoDaya/ASMA-SELD](https://huggingface.co/NutrinoDaya/ASMA-SELD)
- **Published Paper**: [https://zenodo.org/records/17567932](https://zenodo.org/records/17567932)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Ablation Studies](#ablation-studies)
- [Citation](#citation)

## Overview

ASMA-SLED is an advanced neural architecture for simultaneous sound event localization and detection that combines:

- **Phase-aware Audio Processing**: Sophisticated phase encoding for enhanced spatial awareness
- **Visual Context Integration**: CNN-based video feature extraction with ResNet backbone
- **Squeeze-and-Excitation Blocks**: Adaptive channel attention for improved feature representation
- **Transformer-based Fusion**: Advanced multimodal attention mechanism for audio-visual integration
- **Multi-ACCDOA Output**: Activity-Coupled Cartesian Direction of Arrival for precise localization

## Features

### Core Capabilities
- ✅ **Multi-modal Processing**: Joint audio-visual analysis for enhanced performance
- ✅ **Phase-aware Architecture**: Explicit phase encoding for spatial information preservation
- ✅ **Production Ready**: Professional code quality with comprehensive error handling
- ✅ **Configurable Training**: YAML-based configuration system for reproducible experiments
- ✅ **Ablation Framework**: Systematic component analysis for research purposes
- ✅ **Mixed Precision**: Optimized training with automatic mixed precision (AMP)

### Technical Specifications
- **Input**: 4-channel audio (24kHz) + synchronized video (25fps, 224×224)
- **Output**: 13-class sound events with 3D localization (F-score: 21.5%, DOA error: 22.5°)
- **Architecture**: CNN-RNN hybrid with transformer-based multimodal fusion
- **Training**: End-to-end optimization with ADPIT loss function

## Performance

| Model Variant | F-score | DOA Error | Distance Error |
|---------------|---------|-----------|----------------|
| **ASMA-SLED (Full)** | **21.5%** | **22.5°** | **18.3m** |
| ASMA-SLED (No Phase) | 19.2% | 25.1° | 20.7m |
| ASMA-SLED (No SE) | 20.1% | 23.8° | 19.5m |
| ASMA-SLED (No Transformer) | 18.7% | 26.4° | 22.1m |
| Baseline Audio-only | 16.8% | 28.2° | 24.6m |

## Installation

### Prerequisites
```bash
# System requirements
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM
- 50GB+ storage for dataset
```

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd ASMA-SLED

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Pre-trained Checkpoint

**Download our pre-trained model checkpoint:**

🤗 **Hugging Face**: [https://huggingface.co/NutrinoDaya/ASMA-SELD](https://huggingface.co/NutrinoDaya/ASMA-SELD/upload/main)

The checkpoint includes the trained ASMA-SELD model achieving 21.5% F-score on the DCASE 2025 Task 3 development set.

### 2. Dataset Preparation
```bash
# Download DCASE2025 SELD dataset
# Place in: DCASE2025_SELD_dataset/

# Extract features (first time only)
python utils/extract_features.py
```

### 3. Training
```bash
# Train full ASMA-SELD model
python train.py --config configs/ablation/asma_seld_full.yaml

# Train baseline for comparison
python train.py --config configs/baseline_audio.yaml

# Resume training
python train.py --config configs/ablation/asma_seld_full.yaml --resume checkpoints/SELDnet_audio_multiACCDOA_20241107_141639
```

### 4. Inference with Pre-trained Model
```bash
# Run inference using the downloaded checkpoint
python inference.py --config configs/asma_seld_audiovisual.yaml \
    --checkpoint path/to/best_model.pth \
    --output_dir outputs/inference_results
```

### 5. Evaluation
```bash
# Evaluate trained model
python main.py --config configs/ablation/asma_seld_full.yaml --mode evaluate

# Check results
python check_baseline_results.py
```

## Configuration

The system uses YAML configuration files for reproducible experiments:

```yaml
# Example: configs/ablation/asma_seld_full.yaml
net_type: 'asma_seld_ablation'
modality: 'audio_visual'

# ASMA-SLED Components
ablation:
  enable_phase_encoding: true
  enable_se_blocks: true  
  enable_transformer_fusion: true

# Training Parameters
batch_size: 8
nb_epochs: 150
learning_rate: 0.001
```

### Available Configurations

| Config File | Description | Use Case |
|-------------|-------------|----------|
| `asma_seld_full.yaml` | Complete ASMA-SELD model | Main training |
| `asma_seld_no_phase_encoding.yaml` | Without phase encoding | Phase ablation |
| `asma_seld_no_se_blocks.yaml` | Without SE blocks | SE ablation |
| `asma_seld_no_transformer_fusion.yaml` | Without transformer fusion | Fusion ablation |

## Training

### Standard Training
```bash
# Full model training (200 epochs, ~1 hour on RTX 4090)
python train.py --config configs/ablation/asma_seld_full.yaml
```

### Advanced Options
```bash
# Enable mixed precision for faster training
# Edit config: use_amp: true

# Adjust batch size based on GPU memory
# 24GB VRAM: batch_size: 16
# 12GB VRAM: batch_size: 8  
# 8GB VRAM:  batch_size: 4
```

### Monitoring
```bash
# Training logs saved to: logs/
# Checkpoints saved to: checkpoints/
# TensorBoard logs: tensorboard --logdir logs/
```

## Evaluation

### Metrics Computation
The system evaluates performance using standard SELD metrics:

- **F-score**: Sound event detection accuracy
- **DOA Error**: Direction of arrival localization error (degrees)
- **Distance Error**: Source distance estimation error (meters)

### Validation
```bash
# Automatic validation during training (configurable epochs)
eval_epochs: [50, 100, 150]

# Manual evaluation
python main.py --config configs/ablation/asma_seld_full.yaml --mode evaluate
```

## Architecture

### ASMA-SLED Components

```
Audio Input (4ch) ──► Phase Encoder ──► Audio CNN ──► SE Blocks ──┐
                                                                    │
                                                                    ▼
Video Input ──────► Video CNN (ResNet) ─────────────────► Transformer Fusion ──► RNN ──► Multi-ACCDOA Output
```

### Key Innovations

1. **Phase-aware Processing**: Explicit magnitude and phase encoding
2. **SE Blocks**: Channel-wise attention for feature enhancement  
3. **Transformer Fusion**: Cross-modal attention for audio-visual integration
4. **Multi-ACCDOA**: Activity-Coupled Cartesian Direction of Arrival

## Project Structure

```
ASMA-SLED/
├── configs/                    # Configuration files
│   ├── ablation/              # Ablation study configs
│   ├── baseline_audio.yaml    # Audio-only baseline
│   └── asma_seld_audiovisual.yaml
├── models/                    # Model architectures
│   ├── ablation/              # Ablation model variants
│   ├── baseline.py            # Baseline SELDNet
│   └── ASMA_SLED.py          # Main ASMA-SLED model
├── utils/                     # Utilities
│   ├── data_generator.py      # Data loading
│   ├── extract_features.py    # Feature extraction
│   ├── loss.py               # Loss functions
│   └── metrics.py            # Evaluation metrics
├── docs/                      # Documentation
├── paper_output/             # Research paper outputs
├── train.py                  # Main training script
└── main.py                   # Evaluation script
```

## Ablation Studies

Systematic analysis of ASMA-SLED components:

### 1. Phase Encoding Ablation
```bash
python train.py --config configs/ablation/asma_seld_no_phase_encoding.yaml
```

### 2. SE Blocks Ablation  
```bash
python train.py --config configs/ablation/asma_seld_no_se_blocks.yaml
```

### 3. Transformer Fusion Ablation
```bash
python train.py --config configs/ablation/asma_seld_no_transformer_fusion.yaml
```

### Results Analysis
Each ablation demonstrates the contribution of individual components to overall performance, validating the architectural design choices.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{dayarneh2024asma,
  title={ASMA-SLED: Advanced Spatial-aware Multi-modal Attention for Sound Event Localization and Detection},
  author={Dayarneh, Mohammad and others},
  booktitle={Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop},
  year={2024},
  organization={DCASE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Mohammad Dayarneh**  
Email: m.daya.nutrino@gmail.com

---

**Professional Implementation** | **Production Ready** | **Research Validated** | **DCASE 2025**