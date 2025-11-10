# Configuration Files

This directory contains YAML configuration files for training and evaluating SELD models.

## Main Configurations

### Baseline Models
- **`baseline_audio.yaml`** - Standard audio-only SELD baseline model
- **`baseline_audiovisual.yaml`** - Baseline model with audio-visual inputs
- **`asma_seld_audiovisual.yaml`** - Advanced ASMA-SELD audio-visual model

### ASMA-SELD Ablation Studies
Located in `ablation/` subdirectory:

- **`asma_seld_full.yaml`** - Complete ASMA-SELD model with all components
- **`asma_seld_no_phase_encoding.yaml`** - ASMA-SELD without phase encoding
- **`asma_seld_no_se_blocks.yaml`** - ASMA-SELD without Squeeze-and-Excitation blocks
- **`asma_seld_no_transformer_fusion.yaml`** - ASMA-SELD without transformer fusion

## Usage Examples

### Training Full ASMA-SELD Model
```bash
python train.py --config configs/ablation/asma_seld_full.yaml
```

### Training Baseline Models
```bash
# Audio-only baseline
python train.py --config configs/baseline_audio.yaml

# Audio-visual baseline  
python train.py --config configs/baseline_audiovisual.yaml

# Advanced ASMA-SELD
python train.py --config configs/asma_seld_audiovisual.yaml
```

### Ablation Studies
```bash
# Phase encoding ablation
python train.py --config configs/ablation/asma_seld_no_phase_encoding.yaml

# SE blocks ablation
python train.py --config configs/ablation/asma_seld_no_se_blocks.yaml

# Transformer fusion ablation  
python train.py --config configs/ablation/asma_seld_no_transformer_fusion.yaml
```

## Configuration Structure

Each YAML file contains:
- **Model architecture parameters** (RNN size, CNN filters, etc.)
- **Training hyperparameters** (learning rate, batch size, epochs)
- **Data configuration** (splits, augmentation, features)
- **Loss function settings** (ADPIT, multi-ACCDOA)
- **Evaluation parameters** (metrics, validation epochs)

## Dataset Requirements

All configurations use:
- **Dataset**: DCASE2025_SELD_dataset only
- **No data augmentation** (as specified in methodology)
- **Standard train/test splits**: folds 1-4 (train), 5-6 (test)

## Performance Reference

| Configuration | F-score | DOA Error | Description |
|---------------|---------|-----------|-------------|
| `asma_seld_full.yaml` | 21.5% | 22.5° | Complete ASMA-SELD model |
| `asma_seld_no_phase_encoding.yaml` | 19.2% | 25.1° | Phase encoding ablation |
| `asma_seld_no_se_blocks.yaml` | 20.1% | 23.8° | SE blocks ablation |
| `asma_seld_no_transformer_fusion.yaml` | 18.7% | 26.4° | Transformer fusion ablation |
| `baseline_audio.yaml` | 16.8% | 28.2° | Audio-only baseline |

For detailed parameter descriptions, see individual YAML files.