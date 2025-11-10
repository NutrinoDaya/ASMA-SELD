"""
SELD Validation Script

Clean validation script that loads outputs from the outputs folder
and computes SELD metrics without formatting issues.

Usage:
    python validate.py --output_dir outputs/model_audio_20241109_123456
    python validate.py --output_dir outputs/baseline_audio_visual_20241109_123456

Author: Production version for SELD validation
"""

import os
import sys
import argparse
import logging
import glob
import numpy as np

# Add utils to path
sys.path.append('utils')

from metrics import ComputeSELDResults
import utils as seld_utils


def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def detect_modality(output_dir):
    """Detect modality from output directory name."""
    dir_name = os.path.basename(output_dir).lower()
    
    if 'audio_visual' in dir_name or 'audiovisual' in dir_name:
        return 'audio_visual'
    elif 'audio' in dir_name:
        return 'audio'
    else:
        # Default assumption
        return 'audio'


def load_predictions_and_labels(output_dir, data_dir='./DCASE2025_SELD_dataset'):
    """Load predictions and corresponding ground truth labels."""
    logger = logging.getLogger(__name__)
    
    # Find prediction files
    pred_dir = os.path.join(output_dir, 'dev-test')
    if not os.path.exists(pred_dir):
        pred_dir = output_dir
    
    pred_files = glob.glob(os.path.join(pred_dir, '*.csv'))
    
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")
    
    logger.info(f"Found {len(pred_files)} prediction files")
    
    # Load predictions
    pred_dict = {}
    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        pred_dict[filename] = seld_utils.load_output_format_file(pred_file)
    
    # Load corresponding ground truth labels
    label_dict = {}
    label_dir = os.path.join(data_dir, 'metadata_dev')
    
    for filename in pred_dict.keys():
        label_file = os.path.join(label_dir, filename)
        if os.path.exists(label_file):
            label_dict[filename] = seld_utils.load_output_format_file(label_file)
        else:
            logger.warning(f"Ground truth not found for {filename}")
    
    logger.info(f"Loaded {len(label_dict)} ground truth files")
    
    return pred_dict, label_dict


def compute_metrics(pred_dict, label_dict, modality):
    """Compute SELD metrics."""
    logger = logging.getLogger(__name__)
    
    # Setup metrics computation
    score_obj = ComputeSELDResults(
        nb_classes=13,
        doa_threshold=20,
        dist_threshold=float('inf'),
        reldist_threshold=1.0,
        average='macro'
    )
    
    logger.info("Computing SELD metrics...")
    
    # Process each file
    for filename in pred_dict.keys():
        if filename in label_dict:
            pred = pred_dict[filename]
            label = label_dict[filename]
            
            score_obj.update_seld_scores(pred, label)
    
    # Get final scores
    ER, F, LE, LR, seld_scr, classwise_results = score_obj.compute_seld_scores()
    
    return ER, F, LE, LR, seld_scr, classwise_results


def print_results(ER, F, LE, LR, seld_scr, classwise_results, modality):
    """Print validation results cleanly."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"F-score: {F * 100:.1f}%")
    logger.info(f"DOA error: {LE:.1f}°")
    logger.info(f"Distance error: {LR:.1f}m")
    
    if modality == 'audio_visual':
        # Compute onscreen accuracy if available
        if hasattr(classwise_results, 'onscreen_accuracy'):
            logger.info(f"Onscreen accuracy: {classwise_results.onscreen_accuracy * 100:.1f}%")
    
    logger.info("=" * 60)
    
    # Class-wise results
    if classwise_results is not None and len(classwise_results) > 0:
        logger.info("Class-wise F-scores:")
        
        class_names = [
            'alarm', 'baby', 'crash', 'dog', 'engine', 'female', 
            'fire', 'footsteps', 'knock', 'male', 'phone', 'piano', 'speech'
        ]
        
        for idx, class_name in enumerate(class_names):
            if idx < len(classwise_results):
                score = classwise_results[idx]
                if isinstance(score, (int, float)):
                    logger.info(f"  {class_name:12}: {score * 100:5.1f}%")
                else:
                    logger.info(f"  {class_name:12}: {str(score):>5}")


def main():
    parser = argparse.ArgumentParser(description='SELD Validation Script')
    parser.add_argument('--output_dir', required=True,
                       help='Path to model output directory')
    parser.add_argument('--data_dir', default='./DCASE2025_SELD_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--modality', default=None,
                       choices=['audio', 'audio_visual'],
                       help='Modality (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Validate input
    if not os.path.exists(args.output_dir):
        logger.error(f"Output directory not found: {args.output_dir}")
        return 1
    
    # Detect modality if not specified
    if args.modality is None:
        args.modality = detect_modality(args.output_dir)
    
    logger.info(f"Validating output directory: {os.path.basename(args.output_dir)}")
    logger.info(f"Detected modality: {args.modality}")
    
    try:
        # Load predictions and labels
        pred_dict, label_dict = load_predictions_and_labels(args.output_dir, args.data_dir)
        
        # Compute metrics
        ER, F, LE, LR, seld_scr, classwise_results = compute_metrics(pred_dict, label_dict, args.modality)
        
        # Print results
        print_results(ER, F, LE, LR, seld_scr, classwise_results, args.modality)
        
        logger.info("Validation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())