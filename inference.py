"""
SELD Inference Script

Production-ready inference script that loads trained models and generates
predictions on the test set, saving results to the outputs folder.

Usage:
    python inference.py --config configs/baseline_audio.yaml --checkpoint checkpoints/model.pth
    python inference.py --config configs/asma_seld_audiovisual.yaml --checkpoint checkpoints/model.pth

Author: Production version for SELD inference
"""

import os
import sys
import torch
import argparse
import logging
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add utils to path
sys.path.append('utils')

from data_generator import DataGenerator
from extract_features import SELDFeatureExtractor
import utils as seld_utils


def setup_logging():
    """Setup logging for inference."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested structures for easier access
    flattened = {}
    flattened.update(config)
    
    # Flatten model config
    if 'model' in config:
        flattened.update(config['model'])
    
    # Flatten data config
    if 'data' in config:
        flattened.update(config['data'])
    
    # Flatten training config (for batch_size, etc.)
    if 'training' in config:
        flattened.update(config['training'])
    
    return flattened


def load_model(config, checkpoint_path, device):
    """Load trained model from checkpoint using config."""
    logger = logging.getLogger(__name__)
    
    model_type = config.get('net_type', config.get('model_type', 'baseline'))
    modality = config.get('modality', 'audio')
    
    # Import appropriate model
    if model_type == 'baseline':
        from models.baseline import SELDnet
        model = SELDnet(config, modality)
    elif model_type == 'asma_seld_audiovisual':
        from models.ASMA_SELD import ASMA_SELD_AudioVisual
        model = ASMA_SELD_AudioVisual(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully: {model_type} ({modality})")
    return model, config


def run_inference(model, data_loader, config, device, output_dir):
    """Run inference on test data and save predictions."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Running inference, saving to: {output_dir}")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_audio, batch_video, _) in enumerate(tqdm(data_loader, desc="Inference")):
            
            batch_audio = batch_audio.to(device).float()
            if batch_video is not None:
                batch_video = batch_video.to(device).float()
            
            # Forward pass
            if batch_video is not None:
                pred_dict = model(batch_audio, batch_video)
            else:
                pred_dict = model(batch_audio)
            
            # Extract predictions
            if 'multi_accdoa' in pred_dict:
                pred = pred_dict['multi_accdoa']
            else:
                pred = pred_dict
            
            pred_np = pred.cpu().data.numpy()
            
            # Save predictions for each file in batch
            for file_cnt, pred_file in enumerate(pred_np):
                # Get filename from data loader
                filename = data_loader.dataset.get_filename(batch_idx * data_loader.batch_size + file_cnt)
                
                # Convert to required format and save
                output_file = os.path.join(output_dir, filename.replace('.wav', '.csv'))
                
                # Convert multi-ACCDOA to output format
                output_dict = seld_utils.convert_multi_accdoa_to_output_format(
                    pred_file, config['nb_classes'], config['max_polyphony']
                )
                
                seld_utils.write_output_format_file(output_file, output_dict)
    
    logger.info(f"Inference completed. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SELD Inference Script')
    parser.add_argument('--config', required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for inference (overrides config)')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--test_fold', default=None,
                       help='Test fold to use (overrides config, e.g., fold4)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get model info
    model_type = config.get('net_type', config.get('model_type', 'baseline'))
    modality = config.get('modality', 'audio')
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join('outputs', f"{model_type}_{modality}_{timestamp}")
    
    # Load model
    model, config = load_model(config, args.checkpoint, device)
    
    # Setup data generator
    logger.info("Setting up data generator...")
    
    # Get data paths from config or use defaults
    data_dir = config.get('root_dir', './DCASE2025_SELD_dataset')
    feat_dir = config.get('feat_dir', './DCASE2025_SELD_dataset/features')
    
    # Determine test fold
    if args.test_fold:
        test_splits = [args.test_fold]
    else:
        test_splits = config.get('dev_test_folds', ['fold4'])
    
    # Create data generator for test set
    data_gen_test = DataGenerator(
        dataset_dir=data_dir,
        feat_dir=feat_dir,
        mode='test',
        split=test_splits,
        sampling_rate=config.get('sampling_rate', 24000),
        hop_length_s=config.get('hop_length_s', 0.02),
        nb_mels=config.get('nb_mels', 64),
        fps=config.get('fps', 10),
        resnet_feature_size=config.get('resnet_feature_size', 49),
        label_sequence_length=config.get('label_sequence_length', 50),
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        nb_workers=0,
        modality=modality
    )
    
    test_loader = DataLoader(
        dataset=data_gen_test,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Test set size: {len(data_gen_test)} files")
    
    # Run inference
    run_inference(model, test_loader, config, device, args.output_dir)
    
    logger.info("Inference completed successfully!")


if __name__ == '__main__':
    main()