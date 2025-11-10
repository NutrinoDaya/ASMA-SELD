"""
config_loader.py

Configuration loader utility for SELD training pipeline.
Loads YAML configurations and converts them to parameter dictionaries.

Features:
- YAML configuration loading
- Parameter validation
- Environment variable substitution
- Configuration merging and overrides
- Type conversion and validation

Author: Enhanced SELD Pipeline
Date: October 29, 2025
"""

import os
import yaml
from typing import Dict, Any, Union
from datetime import datetime
import torch


class ConfigLoader:
    """Load and validate YAML configurations for SELD training."""
    
    def __init__(self, config_path: str):
        """
        Initialize config loader.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load YAML configuration from file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def to_params(self) -> Dict[str, Any]:
        """
        Convert YAML config to flat parameter dictionary compatible with existing code.
        
        Returns:
            dict: Flattened parameter dictionary
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        
        params = {}
        
        # Model parameters
        model_config = self.config.get('model', {})
        params.update({
            'modality': self.config.get('modality', 'audio'),  # Top-level modality
            'net_type': 'SELDnet',
            
            # Architecture
            'nb_conv_blocks': model_config.get('nb_conv_blocks', 3),
            'nb_conv_filters': model_config.get('nb_conv_filters', 64),
            'f_pool_size': model_config.get('f_pool_size', [4, 4, 2]),
            't_pool_size': model_config.get('t_pool_size', [5, 1, 1]),
            'dropout': model_config.get('dropout', 0.05),
            
            'rnn_size': model_config.get('rnn_size', 128),
            'nb_rnn_layers': model_config.get('nb_rnn_layers', 2),
            
            'nb_self_attn_layers': model_config.get('nb_self_attn_layers', 2),
            'nb_attn_heads': model_config.get('nb_attn_heads', 8),
            'nb_transformer_layers': model_config.get('nb_transformer_layers', 2),
            
            'nb_fnn_layers': model_config.get('nb_fnn_layers', 1),
            'fnn_size': model_config.get('fnn_size', 128),
            
            'max_polyphony': model_config.get('max_polyphony', 3),
            'nb_classes': model_config.get('nb_classes', 13),
            'label_sequence_length': model_config.get('label_sequence_length', 50),
            'multiACCDOA': model_config.get('multiACCDOA', True),
            'thresh_unify': model_config.get('thresh_unify', 15),
        })
        
        # Training parameters - check both training section and top-level
        training_config = self.config.get('training', {})
        params.update({
            'nb_epochs': training_config.get('nb_epochs', self.config.get('nb_epochs', 200)),
            'learning_rate': training_config.get('learning_rate', self.config.get('learning_rate', 0.001)),
            'weight_decay': training_config.get('weight_decay', self.config.get('weight_decay', 0)),
            'batch_size': training_config.get('batch_size', self.config.get('batch_size', 256)),
            'eval_frequency': training_config.get('eval_frequency', self.config.get('eval_frequency', 25)),
            'save_frequency': training_config.get('save_frequency', self.config.get('save_frequency', 50)),
            'nb_workers': training_config.get('nb_workers', self.config.get('nb_workers', 8)),
            'shuffle': training_config.get('shuffle', self.config.get('shuffle', True)),
        })
        
        # Data parameters
        data_config = self.config.get('data', {})
        params.update({
            'root_dir': data_config.get('root_dir', './DCASE2025_SELD_dataset'),
            'feat_dir': data_config.get('feat_dir', './DCASE2025_SELD_dataset/features'),
            'dev_train_folds': data_config.get('dev_train_folds', ['fold2', 'fold3']),
            'dev_test_folds': data_config.get('dev_test_folds', ['fold4']),
            'sampling_rate': data_config.get('sampling_rate', 24000),
            'hop_length_s': data_config.get('hop_length_s', 0.02),
            'nb_mels': data_config.get('nb_mels', 64),
            'fps': data_config.get('fps', 10),
            'resnet_feature_size': data_config.get('resnet_feature_size', 49),
        })
        
        # Augmentation parameters
        aug_config = self.config.get('augmentation', {})
        params.update({
            'use_specaugment': aug_config.get('use_specaugment', False),
            'spec_time_mask_param': aug_config.get('spec_time_mask_param', 30),
            'spec_freq_mask_param': aug_config.get('spec_freq_mask_param', 8),
            'spec_num_time_masks': aug_config.get('spec_num_time_masks', 1),
            'spec_num_freq_masks': aug_config.get('spec_num_freq_masks', 1),
        })
        
        # Metrics parameters
        metrics_config = self.config.get('metrics', {})
        params.update({
            'average': metrics_config.get('average', 'macro'),
            'segment_based_metrics': metrics_config.get('segment_based_metrics', False),
            'lad_doa_thresh': metrics_config.get('lad_doa_thresh', 20),
            'lad_dist_thresh': float(metrics_config.get('lad_dist_thresh', 'inf')),
            'lad_reldist_thresh': float(metrics_config.get('lad_reldist_thresh', 1.0)),
            'lad_req_onscreen': metrics_config.get('lad_req_onscreen', False),
            'use_jackknife': metrics_config.get('use_jackknife', False),
        })
        
        # Logging parameters
        logging_config = self.config.get('logging', {})
        params.update({
            'log_dir': logging_config.get('log_dir', 'logs'),
            'checkpoints_dir': logging_config.get('checkpoints_dir', 'checkpoints'),
            'output_dir': logging_config.get('output_dir', 'outputs'),
        })
        
        return params
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        validation_config = self.config.get('validation', {})
        
        if not validation_config.get('validate_params', True):
            return True
        
        params = self.to_params()
        
        # Validate required parameters
        required_params = [
            'rnn_size', 'fnn_size', 'nb_conv_filters', 'dropout'
        ]
        
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate specific values (from lessons learned)
        validations = {
            'rnn_size': validation_config.get('required_rnn_size', 128),
            'fnn_size': validation_config.get('required_fnn_size', 128),
            'nb_conv_filters': validation_config.get('required_conv_filters', 64),
            'dropout': validation_config.get('required_dropout', 0.05),
        }
        
        for param, expected_value in validations.items():
            if params.get(param) != expected_value:
                raise ValueError(
                    f"Parameter {param}={params.get(param)} does not match "
                    f"required value {expected_value} (from lessons learned)"
                )
        
        return True
    
    def generate_experiment_name(self) -> str:
        """
        Generate unique experiment name based on configuration.
        
        Returns:
            str: Unique experiment name with timestamp
        """
        logging_config = self.config.get('logging', {})
        prefix = logging_config.get('experiment_prefix', 'SELDnet_v85_focused')
        timestamp_format = logging_config.get('timestamp_format', '%Y%m%d_%H%M%S')
        timestamp = datetime.now().strftime(timestamp_format)
        
        return f"{prefix}_{timestamp}"
    
    def get_model_type(self) -> str:
        """Get model type from configuration."""
        return self.config.get('model', {}).get('type', 'v85_focused')
    
    def get_device(self) -> torch.device:
        """
        Get computing device from configuration.
        
        Returns:
            torch.device: Configured device
        """
        hardware_config = self.config.get('hardware', {})
        device_config = hardware_config.get('device', 'auto')
        
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device_config)
    
    def print_summary(self):
        """Print configuration summary."""
        print("="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})
        aug_config = self.config.get('augmentation', {})
        
        print(f"Model Type: {model_config.get('type', 'unknown')}")
        print(f"Modality: {model_config.get('modality', 'unknown')}")
        print(f"Architecture: {model_config.get('rnn_size', '?')}-RNN, {model_config.get('fnn_size', '?')}-FNN")
        print(f"Parameters: ~{model_config.get('rnn_size', 128) * 6000 + 30000:,} (estimated)")
        print(f"Epochs: {training_config.get('nb_epochs', '?')}")
        print(f"Learning Rate: {training_config.get('learning_rate', '?')}")
        print(f"Dropout: {model_config.get('dropout', '?')}")
        print(f"Batch Size: {training_config.get('batch_size', '?')}")
        print(f"SpecAugment: {aug_config.get('use_specaugment', False)}")
        
        if aug_config.get('use_specaugment', False):
            print(f"   - Time mask: {aug_config.get('spec_time_mask_param', '?')}")
            print(f"   - Freq mask: {aug_config.get('spec_freq_mask_param', '?')}")
        
        print("="*80)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load and validate configuration.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Validated parameter dictionary
    """
    loader = ConfigLoader(config_path)
    loader.validate_config()
    return loader.to_params()


if __name__ == "__main__":
    # Test configuration loading
    config_path = "configs/v85_focused_config.yaml"
    
    try:
        loader = ConfigLoader(config_path)
        loader.print_summary()
        
        params = loader.to_params()
        print(f"\nINFO: Configuration loaded: {len(params)} parameters")
        print(f"Experiment name: {loader.generate_experiment_name()}")
        
    except Exception as e:
        print(f"ERROR: Configuration loading failed: {e}")