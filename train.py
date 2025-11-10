"""
Professional SELD Training Script

Production-ready training script for Sound Event Localization and Detection models.
Supports YAML configuration files for reproducible experiments.

Supported Models:
    - baseline: Standard SELD baseline model  
    - asma_seld_audiovisual: Advanced ASMA-SELD model with audio-visual fusion
    - asma_seld_ablation: ASMA-SELD variants for ablation studies

Usage:
    python train.py --config configs/baseline_audio.yaml
    python train.py --config configs/asma_seld_audiovisual.yaml
    python train.py --config configs/ablation/asma_seld_full.yaml

Author: Mohammad Dayarneh
Email: m.daya.nutrino@gmail.com
"""

import os
import sys
import torch
import yaml
import argparse
import logging
import time
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import utilities
sys.path.append('utils')

from utils.loss import SELDLossADPIT, SELDLossSingleACCDOA
from utils.loss_balanced import SELDLossADPITBalanced
from utils.metrics import ComputeSELDResults
from utils.data_generator import DataGenerator
from utils.extract_features import SELDFeatureExtractor
import utils as seld_utils


def setup_logging(log_dir, model_name):
    """Setup logging to file and console with professional formatting."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {model_name} training session")
    logger.info(f"Log file: {log_file}")
    return logger


def create_experiment_directories(model_name, modality):
    """Create experiment directories with systematic naming convention."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_name}_{modality}_{timestamp}"
    
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    output_dir = os.path.join('outputs', exp_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return checkpoint_dir, output_dir, exp_name


def load_config(config_path):
    """Load and flatten YAML configuration for compatibility with legacy utils."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested structures for compatibility
    flattened = {}
    flattened.update(config)
    
    # Flatten nested sections
    for section in ['audio', 'video', 'model', 'training', 'optimizer', 'loss', 'data', 'metrics']:
        if section in config:
            for key, value in config[section].items():
                flattened[key] = value
    
    return flattened


def custom_collate_fn(batch):
    """Custom collate function ensuring CPU tensor handling during batching."""
    if isinstance(batch[0][0], tuple):
        # Audio-visual modality: ((audio, video), labels)
        audio_list = [item[0][0].cpu() for item in batch]
        video_list = [item[0][1].cpu() for item in batch]
        labels_list = [item[1].cpu() for item in batch]
        
        audio_batch = torch.stack(audio_list, 0)
        video_batch = torch.stack(video_list, 0)
        labels_batch = torch.stack(labels_list, 0)
        
        return (audio_batch, video_batch), labels_batch
    else:
        # Audio-only modality: (audio, labels)
        audio_list = [item[0].cpu() for item in batch]
        labels_list = [item[1].cpu() for item in batch]
        
        audio_batch = torch.stack(audio_list, 0)
        labels_batch = torch.stack(labels_list, 0)
        
        return audio_batch, labels_batch


def create_model(model_type, config):
    """Factory function for model creation with proper error handling."""
    try:
        if model_type == 'baseline':
            from models.baseline import SELDNet
            return SELDNet(config)
        elif model_type == 'asma_seld_audiovisual':
            from models.ASMA_SELD import ASMA_SELD_AudioVisual
            return ASMA_SELD_AudioVisual(config)
        elif model_type == 'asma_seld_ablation':
            # Determine ablation variant from config
            ablation_config = config.get('ablation', {})
            if not ablation_config.get('enable_phase_encoding', True):
                from models.ablation.asma_sled_no_phase import ASMASLEDNoPhase
                return ASMASLEDNoPhase(config)
            elif not ablation_config.get('enable_se_blocks', True):
                from models.ablation.asma_sled_no_se import ASMASLEDNoSE
                return ASMASLEDNoSE(config)
            elif not ablation_config.get('enable_transformer_fusion', True):
                from models.ablation.asma_sled_no_transformer import ASMASLEDNoTransformer
                return ASMASLEDNoTransformer(config)
            else:
                from models.ASMA_SELD import ASMA_SELD_AudioVisual
                return ASMA_SELD_AudioVisual(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except ImportError as e:
        raise ImportError(f"Failed to import model {model_type}: {e}")


def initialize_seld_metrics(config):
    """Initialize SELD metrics computer with deferred loading for startup optimization."""
    global seld_metric_computer
    seld_metric_computer = None
    
    try:
        print("INFO: Initializing SELD metrics computer...")
        
        # Setup metrics computation
        use_jackknife = config.get('use_jackknife', False)
        average = config.get('average', 'macro')
        
        seld_metric_computer = ComputeSELDResults(
            average=average,
            segment_based_metrics=config.get('segment_based_metrics', False),
            lad_doa_thresh=config.get('lad_doa_thresh', 20),
            lad_dist_thresh=config.get('lad_dist_thresh', float('inf')),
            lad_reldist_thresh=config.get('lad_reldist_thresh', 1.0),
            lad_req_onscreen=config.get('lad_req_onscreen', False),
            use_jackknife=use_jackknife
        )
        
        if seld_metric_computer:
            print("INFO: SELD metrics computer initialized successfully")
        else:
            print("WARNING: Failed to initialize SELD metrics computer")
            
    except Exception as e:
        print(f"ERROR: Failed to initialize SELD metrics: {e}")


def compute_seld_metrics(predictions, ground_truth, config):
    """Compute SELD metrics with error handling."""
    print("INFO: Computing SELD evaluation metrics...")
    
    try:
        if seld_metric_computer is None:
            print("WARNING: SELD metrics computer not available")
            return None, None, None
        
        return seld_metric_computer.get_SELD_Results(predictions, ground_truth)
        
    except Exception as e:
        print(f"ERROR: SELD metrics computation failed: {e}")
        return None, None, None


def main():
    """Main training function with comprehensive configuration and error handling."""
    parser = argparse.ArgumentParser(description='SELD Model Training')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser.add_argument('--resume', default=None, help='Path to checkpoint directory for resuming')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device and model parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = config.get('net_type', 'baseline')
    modality = config.get('modality', 'audio')
    
    # Display configuration
    print(f'Device: {device}')
    print(f'Model: {model_type}')
    print(f'Modality: {modality}')
    print(f'Batch size: {config["batch_size"]}')
    print(f'Workers: {config.get("nb_workers", 0)}')
    print(f'Epochs: {config["nb_epochs"]}')
    
    # Create experiment directories
    checkpoint_dir, output_dir, exp_name = create_experiment_directories(model_type, modality)
    
    # Setup logging
    logger = setup_logging('logs', exp_name)
    
    # Setup feature extractor and data generators
    print('INFO: Setting up data loaders...')
    
    feat_extractor = SELDFeatureExtractor(config)
    
    train_generator = DataGenerator(
        config=config,
        split=config['dev_train_folds'],
        feat_extractor=feat_extractor
    )
    
    test_generator = DataGenerator(
        config=config,
        split=config['dev_test_folds'],
        feat_extractor=feat_extractor
    )
    
    # Configure data loaders with optimized parameters
    num_workers = config.get('nb_workers', 0)
    val_batch_size = min(64, config['batch_size'])  # Optimize validation batch size
    val_workers = min(4, num_workers) if num_workers > 0 else 0
    
    dev_train_iterator = DataLoader(
        dataset=train_generator,
        batch_size=config['batch_size'],
        shuffle=config.get('shuffle', True),
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    dev_test_iterator = DataLoader(
        dataset=test_generator,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f'Training batches: {len(dev_train_iterator)}')
    print(f'Validation batches: {len(dev_test_iterator)} (batch_size: {val_batch_size})')
    
    # Create model
    model = create_model(model_type, config)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model created: {param_count:,} trainable parameters')
    
    # Setup loss function
    if config.get('multiACCDOA', False):
        loss_fn = SELDLossADPITBalanced(config)
    else:
        loss_fn = SELDLossSingleACCDOA(config)
    
    loss_fn = loss_fn.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0)
    )
    
    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', False) and device == 'cuda' else None
    
    # Defer SELD metrics setup for faster startup
    print('INFO: Deferring SELD metrics setup for optimized startup...')
    
    # Resume training if specified
    start_epoch = 0
    best_f_score = 0.0
    
    if args.resume:
        try:
            checkpoint_path = os.path.join(args.resume, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_f_score = checkpoint.get('best_f_score', 0.0)
            
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            print(f'INFO: Resumed from epoch {start_epoch}, best F-score: {best_f_score * 100:.1f}%')
        except Exception as e:
            print(f'ERROR: Failed to load checkpoint: {e}')
    
    print(f'\nINFO: Starting training for {config["nb_epochs"]} epochs...\n')
    
    # Training loop
    for epoch in range(start_epoch, config['nb_epochs']):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dev_train_iterator, desc=f'Epoch {epoch + 1}/{config["nb_epochs"]}')
        
        for batch_idx, (audio_input, target_output) in enumerate(progress_bar):
            try:
                # Move data to device
                if isinstance(audio_input, tuple):
                    audio_feat = audio_input[0].to(device, non_blocking=True)
                    vid_feat = audio_input[1].to(device, non_blocking=True)
                    model_input = (audio_feat, vid_feat)
                else:
                    audio_feat = audio_input.to(device, non_blocking=True)
                    model_input = audio_feat
                    vid_feat = None
                
                target_output = target_output.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if scaler:
                    with torch.cuda.amp.autocast():
                        if vid_feat is not None:
                            model_output = model(audio_feat, vid_feat)
                        else:
                            model_output = model(audio_feat)
                        loss = loss_fn(model_output, target_output)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if vid_feat is not None:
                        model_output = model(audio_feat, vid_feat)
                    else:
                        model_output = model(audio_feat)
                    loss = loss_fn(model_output, target_output)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                avg_loss = train_loss / num_batches
                progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
                
            except Exception as e:
                print(f'ERROR: Training step failed at batch {batch_idx}: {e}')
                continue
        
        avg_train_loss = train_loss / max(num_batches, 1)
        
        # Validation phase
        should_validate = (epoch + 1) in config.get('eval_epochs', [config['nb_epochs']])
        
        if should_validate:
            # Initialize SELD metrics if not done yet
            if 'seld_metric_computer' not in globals() or seld_metric_computer is None:
                initialize_seld_metrics(config)
            
            model.eval()
            val_loss = 0.0
            val_batches = 0
            predictions = []
            ground_truth = []
            
            with torch.no_grad():
                for audio_input, target_output in tqdm(dev_test_iterator, desc='Validation', leave=False):
                    try:
                        # Move data to device
                        if isinstance(audio_input, tuple):
                            audio_feat = audio_input[0].to(device, non_blocking=True)
                            vid_feat = audio_input[1].to(device, non_blocking=True)
                        else:
                            audio_feat = audio_input.to(device, non_blocking=True)
                            vid_feat = None
                        
                        target_output = target_output.to(device, non_blocking=True)
                        
                        # Forward pass
                        if vid_feat is not None:
                            model_output = model(audio_feat, vid_feat)
                        else:
                            model_output = model(audio_feat)
                        
                        loss = loss_fn(model_output, target_output)
                        val_loss += loss.item()
                        val_batches += 1
                        
                        # Collect predictions and ground truth
                        predictions.append(model_output.cpu())
                        ground_truth.append(target_output.cpu())
                        
                    except Exception as e:
                        print(f'ERROR: Validation step failed: {e}')
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            
            # Compute SELD metrics
            if predictions and ground_truth:
                pred_tensor = torch.cat(predictions, dim=0)
                gt_tensor = torch.cat(ground_truth, dim=0)
                
                val_f, val_ang_error, val_dist_error = compute_seld_metrics(
                    pred_tensor.numpy(), gt_tensor.numpy(), config
                )
                
                if val_f is not None:
                    # Combined score for model selection
                    combined_score = val_f * 100 - val_ang_error * 0.5
                    
                    # Save best model
                    if val_f > best_f_score:
                        best_f_score = val_f
                        
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_f_score': best_f_score,
                            'val_loss': avg_val_loss,
                            'val_f_score': val_f,
                            'val_ang_error': val_ang_error,
                            'val_dist_error': val_dist_error,
                            'config': config
                        }
                        
                        if scaler:
                            checkpoint['scaler_state_dict'] = scaler.state_dict()
                        
                        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
                        tqdm.write(f"INFO: New best F-score: {val_f * 100:.1f}% (DOA: {val_ang_error:.1f}°, Dist: {val_dist_error:.1f}) - model saved")
                    
                    tqdm.write(f"Epoch {epoch + 1}: Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | F-score {val_f * 100:.1f}% | DOA {val_ang_error:.1f}° | Dist {val_dist_error:.1f}")
                else:
                    tqdm.write(f"Epoch {epoch + 1}: Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Metrics computation failed")
            else:
                tqdm.write(f"Epoch {epoch + 1}: Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | No predictions collected")
        else:
            tqdm.write(f"Epoch {epoch + 1}: Loss {avg_train_loss:.4f}")
        
        # Save checkpoint at specified intervals
        if (epoch + 1) % config.get('save_frequency', 10) == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pth'
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f_score': best_f_score,
                'train_loss': avg_train_loss,
                'config': config
            }
            
            if scaler:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_name))
            tqdm.write(f"INFO: Checkpoint saved at epoch {epoch + 1} ({checkpoint_name})")
    
    # Training completion
    print('\nINFO: Training completed successfully!')
    print(f'Best F-score achieved: {best_f_score * 100:.1f}%')
    print(f'Checkpoints saved in: {checkpoint_dir}')


if __name__ == '__main__':
    main()