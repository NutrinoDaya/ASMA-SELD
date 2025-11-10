"""
extract_features.py

This module defines the SELDFeatureExtractor class, which provides functionality to extract features from both audio
and video data. It also processes the labels to support MultiACCDOA (ADPIT).  It includes the following key components:

Classes:
    SELDFeatureExtractor: A class that supports the extraction of audio and video features. It extracts log Mel
    spectrogram from audio files and ResNet-based features from video frames. It also processes labels for MultiACCDOA.

    Methods:
        - extract_audio_features: Extracts audio features from a specified split of the dataset.
        - extract_video_features: Extracts video features from a specified split of the dataset.
        - extract_features: A high-level function to extract features based on the modality ('audio' or 'audio_visual').
        - extract_labels: converts labels to support multiACCDOA.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
import glob
import torch
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import utils
from multiprocessing import Pool, cpu_count
from functools import partial


class SELDFeatureExtractor():
    def __init__(self, params):
        """
        Initializes the SELDFeatureExtractor with the provided parameters.
        Args:
            params (dict): A dictionary containing various parameters for audio/video feature extraction among others.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.params = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']

        self.modality = params['modality']

        # audio feature extraction
        self.sampling_rate = params['sampling_rate']
        self.hop_length = int(self.sampling_rate * params['hop_length_s'])
        self.win_length = 2 * self.hop_length
        self.n_fft = 2 ** (self.win_length - 1).bit_length()
        self.nb_mels = params['nb_mels']

        # video feature extraction
        if self.modality == 'audio_visual':
            self.fps = params['fps']
            # Initialize ResNet model and set to evaluation mode
            self.weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=self.weights).to(self.device)
            self.backbone = torch.nn.Sequential(*(list(self.model.children())[:-2]))
            self.backbone.eval()
            
            # Enable mixed precision for faster GPU processing
            if self.device == 'cuda':
                self.backbone = self.backbone.half()  # Use FP16
            
            self.preprocess = self.weights.transforms()

        # label extraction
        self.nb_label_frames = params['label_sequence_length']
        self.nb_unique_classes = params['nb_classes']

    def _process_single_audio(self, audio_file, split):
        """Process a single audio file (for multiprocessing)"""
        filename = os.path.splitext(os.path.basename(audio_file))[0] + '.pt'
        feature_path = os.path.join(self.feat_dir, f'stereo_{split}', filename)
        
        # Check if the feature file already exists
        if os.path.exists(feature_path):
            return None  # Skip existing files
        
        # Extract features
        audio, sr = utils.load_audio(audio_file, self.sampling_rate)
        audio_feat = utils.extract_log_mel_spectrogram(audio, sr, self.n_fft, self.hop_length, self.win_length, self.nb_mels)
        audio_feat = torch.tensor(audio_feat, dtype=torch.float32)
        torch.save(audio_feat, feature_path)
        return filename

    def extract_audio_features(self, split):
        """
        Extracts audio features for a given split (dev/eval) using multiprocessing.
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """

        if split == 'dev':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_dev', 'dev-*', '*.wav'))
        elif split == 'eval':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_eval', 'eval', '*.wav'))
        else:
            raise ValueError("Split must be either 'dev' or 'eval'.")

        os.makedirs(os.path.join(self.feat_dir, f'stereo_{split}'), exist_ok=True)
        
        # Filter out already processed audio files
        remaining_audio = []
        for audio_file in audio_files:
            filename = os.path.splitext(os.path.basename(audio_file))[0] + '.pt'
            feature_path = os.path.join(self.feat_dir, f'stereo_{split}', filename)
            if not os.path.exists(feature_path):
                remaining_audio.append(audio_file)
        
        if not remaining_audio:
            print(f"All audio features already extracted for {split}!")
            return
        
        # Use multiprocessing for faster extraction
        num_workers = min(cpu_count(), 8)  # Limit to 8 workers
        print(f"Extracting {len(remaining_audio)} remaining audio features using {num_workers} workers...")
        
        process_func = partial(self._process_single_audio, split=split)
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, remaining_audio),
                total=len(remaining_audio),
                desc=f"Processing audio files ({split})",
                unit="file"
            ))

    def _process_single_video(self, video_file, split):
        """Process a single video file (no multiprocessing - sequential but GPU optimized)"""
        filename = os.path.splitext(os.path.basename(video_file))[0] + '.pt'
        feature_path = os.path.join(self.feat_dir, f'video_{split}', filename)
        
        # Check if the feature file already exists
        if os.path.exists(feature_path):
            return None  # Skip existing files
        
        try:
            # Load video frames
            video_frames = utils.load_video(video_file, self.fps)
            
            # Extract features on GPU with large batch size
            with torch.no_grad():
                video_feat = utils.extract_resnet_features(video_frames, self.preprocess, self.backbone, self.device, batch_size=128)
                torch.save(video_feat, feature_path)
            
            return filename
        except Exception as e:
            print(f"\nERROR: Error processing {os.path.basename(video_file)}: {e}")
            return None

    def extract_video_features(self, split):
        """
        Extracts video features for a given split (dev/eval) with GPU optimization.
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """
        if split == 'dev':
            video_files = glob.glob(os.path.join(self.root_dir, 'video_dev', 'dev-*', '*.mp4'))
        elif split == 'eval':
            video_files = glob.glob(os.path.join(self.root_dir, 'video_eval', 'eval', '*.mp4'))
        else:
            raise ValueError("Split must be either 'dev' or 'eval'.")

        os.makedirs(os.path.join(self.feat_dir, f'video_{split}'), exist_ok=True)

        # Filter out already processed videos
        remaining_videos = []
        for video_file in video_files:
            filename = os.path.splitext(os.path.basename(video_file))[0] + '.pt'
            feature_path = os.path.join(self.feat_dir, f'video_{split}', filename)
            if not os.path.exists(feature_path):
                remaining_videos.append(video_file)
        
        if not remaining_videos:
            print(f"All video features already extracted for {split}!")
            return
        
        print(f"Processing {len(remaining_videos)} videos with FP16 GPU acceleration (batch_size=128)")
        
        # Sequential processing but GPU-optimized
        for video_file in tqdm(remaining_videos, desc=f"Processing video files ({split})", unit="file"):
            self._process_single_video(video_file, split)

    def extract_features(self, split='dev'):
        """
        Extracts features based on the selected modality ('audio' or 'audio_visual').
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """

        os.makedirs(self.feat_dir, exist_ok=True)

        if self.modality == 'audio':
            self.extract_audio_features(split)
        elif self.modality == 'audio_visual':
            self.extract_audio_features(split)
            self.extract_video_features(split)
        else:
            raise ValueError("Modality should be one of 'audio' or 'audio_visual'. You can set the modality in params.py")

    def _process_single_label(self, label_file, split):
        """Process a single label file (for multiprocessing)"""
        filename = os.path.splitext(os.path.basename(label_file))[0] + '.pt'
        label_path = os.path.join(self.feat_dir, 'metadata_{}{}'.format(split, '_adpit' if self.params['multiACCDOA'] else ''), filename)

        # Check if the feature file already exists
        if os.path.exists(label_path):
            return None

        # If the feature file doesn't exist, perform extraction
        label_data = utils.load_labels(label_file)
        if self.params['multiACCDOA']:
            processed_labels = utils.process_labels_adpit(label_data, self.nb_label_frames, self.nb_unique_classes)
        else:
            processed_labels = utils.process_labels(label_data, self.nb_label_frames, self.nb_unique_classes)
        torch.save(processed_labels, label_path)
        return filename

    def extract_labels(self, split):

        os.makedirs(self.feat_dir, exist_ok=True)   # already created by extract_features method

        if split == 'dev':
            label_files = glob.glob(os.path.join(self.root_dir, 'metadata_dev', 'dev-*', '*.csv'))
        elif split == 'eval':  # only for organizers
            label_files = glob.glob(os.path.join(self.root_dir, 'metadata_eval', 'eval', '*.csv'))
        else:
            raise ValueError("Split must be either 'dev' or 'eval'.")

        os.makedirs(os.path.join(self.feat_dir, 'metadata_{}{}'.format(split, '_adpit' if self.params['multiACCDOA'] else '')), exist_ok=True)

        # Filter out already processed label files
        remaining_labels = []
        for label_file in label_files:
            filename = os.path.splitext(os.path.basename(label_file))[0] + '.pt'
            label_path = os.path.join(self.feat_dir, 'metadata_{}{}'.format(split, '_adpit' if self.params['multiACCDOA'] else ''), filename)
            if not os.path.exists(label_path):
                remaining_labels.append(label_file)
        
        if not remaining_labels:
            print(f"All labels already extracted for {split}!")
            return
        
        # Use multiprocessing for faster extraction
        num_workers = min(cpu_count(), 8)
        print(f"Extracting {len(remaining_labels)} remaining labels using {num_workers} workers...")
        
        process_func = partial(self._process_single_label, split=split)
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, remaining_labels),
                total=len(remaining_labels),
                desc=f"Processing label files ({split})",
                unit="file"
            ))


if __name__ == '__main__':
    # use this space to test if the SELDFeatureExtractor class works as expected.
    # All the classes will be called from the main.py for actual use.
    from parameters import params
    
    # Set multiprocessing start method for Windows
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass
    
    print("="*80)
    print("FAST FEATURE EXTRACTION WITH MULTIPROCESSING")
    print("="*80)
    
    feature_extractor = SELDFeatureExtractor(params)
    
    print("\n1. Extracting audio features...")
    feature_extractor.extract_features(split='dev')
    
    print("\n2. Extracting labels...")
    feature_extractor.extract_labels(split='dev')
    
    print("\nINFO: Feature extraction completed!")
    print("="*80)


