#!/usr/bin/env python3
"""
ASMA-SLED: Audio-Spatial Multi-Attention SELD Network

An advanced Sound Event Localization and Detection model that combines
audio and visual modalities through transformer-based attention mechanisms.

Key Features:
1. Multi-attention audio processing with SE blocks and phase encoding
2. Cross-modal transformer fusion for audio-visual integration
3. Spatial awareness for enhanced localization accuracy
4. Multi-track prediction with onscreen/offscreen classification

Author: Mohammad Dayarneh
Date: November 9, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block - V3's proven component.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEConvBlock(nn.Module):
    """
    V3's proven SE-enhanced convolutional block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=(5, 4), dropout=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.se(x)  
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ASMA_SELD_AudioVisual(nn.Module):
    """
    ASMA-SELD Audio-Visual: F-score champion extended for visual fusion
    
    Architecture:
    1. Audio path: Advanced SE blocks + phase encoding
    2. Visual path: Transformer decoder fusion
    3. Output: Includes onscreen/offscreen prediction
    
    Target: ~2.5M parameters
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # === AUDIO PATH (V3's proven architecture) ===
        
        # Phase encoder for stereo spatial cues
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4)
        )
        
        # Conv layers with SE blocks
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(params['nb_conv_blocks']):
            in_channels = params['nb_conv_filters'] if conv_cnt > 0 else 6  # 2 stereo + 4 phase
            self.conv_blocks.append(SEConvBlock(
                in_channels=in_channels,
                out_channels=params['nb_conv_filters'],
                pool_size=(params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]),
                dropout=params['dropout']
            ))

        # GRU for temporal modeling
        self.gru_input_dim = params['nb_conv_filters'] * int(np.floor(params['nb_mels'] / np.prod(params['f_pool_size'])))
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=params['rnn_size'],
            num_layers=params['nb_rnn_layers'],
            batch_first=True,
            dropout=params['dropout'],
            bidirectional=True
        )

        # Self-attention layers
        self.mhsa_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=params['rnn_size'],
                num_heads=params['nb_attn_heads'],
                dropout=params['dropout'],
                batch_first=True
            ) for _ in range(params['nb_self_attn_layers'])
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(params['rnn_size']) 
            for _ in range(params['nb_self_attn_layers'])
        ])

        # === VISUAL FUSION PATH (for audio-visual mode) ===
        if params['modality'] == 'audio_visual':
            # Visual embedding: 49 (7x7) -> rnn_size
            self.visual_embed_to_d_model = nn.Linear(
                in_features=params['resnet_feature_size'], 
                out_features=params['rnn_size']
            )
            
            # Transformer decoder for audio-visual fusion
            # Tuned to reach ~2.5M params
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(
                d_model=params['rnn_size'],
                nhead=params['nb_attn_heads'],
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(
                self.transformer_decoder_layer,
                num_layers=params.get('nb_transformer_layers', 2)  # 2 layers for ~2.5M
            )

        # === OUTPUT PATH ===
        
        # FFN layers
        self.fnn_list = nn.ModuleList()
        for fc_cnt in range(params['nb_fnn_layers']):
            self.fnn_list.append(
                nn.Linear(
                    params['fnn_size'] if fc_cnt else params['rnn_size'],
                    params['fnn_size'],
                    bias=True
                )
            )

        # Output dimension
        if params['modality'] == 'audio_visual':
            if params['multiACCDOA']:
                # 4 => (x,y), distance, on/off screen
                self.output_dim = params['max_polyphony'] * 4 * params['nb_classes']
            else:
                self.output_dim = 4 * params['nb_classes']
        else:
            if params['multiACCDOA']:
                self.output_dim = params['max_polyphony'] * 3 * params['nb_classes']
            else:
                self.output_dim = 3 * params['nb_classes']
                
        self.fnn_list.append(
            nn.Linear(
                params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'],
                self.output_dim,
                bias=True
            )
        )

        # Activations
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        if params['modality'] == 'audio_visual':
            self.onscreen_act = nn.Sigmoid()

    def forward(self, audio_feat, vid_feat=None):
        """
        Forward pass for ASMA-SELD Audio-Visual.
        
        Args:
            audio_feat: [B, 2, 251, 64] - stereo spectrogram
            vid_feat: [B, 50, 7, 7] - visual feature map (optional)
            
        Returns:
            pred: [B, 50, 117] for audio-only multiACCDOA
                  [B, 50, 156] for audio-visual multiACCDOA
        """
        # === AUDIO PROCESSING (V3's proven path) ===
        
        # Phase encoding
        phase_diff = audio_feat[:, 0:1, :, :] - audio_feat[:, 1:2, :, :]
        phase_feat = F.relu(self.phase_encoder(phase_diff))
        x = torch.cat([audio_feat, phase_feat], dim=1)  # [B, 6, T, F]

        # SE-enhanced convolution
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Reshape for temporal processing
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        # GRU temporal modeling
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]

        # Self-attention
        for mhsa, ln in zip(self.mhsa_layers, self.layer_norms):
            x_in = x
            x, _ = mhsa(x_in, x_in, x_in)
            x = x + x_in
            x = ln(x)

        # === VISUAL FUSION (if visual features provided) ===
        
        if vid_feat is not None and self.params['modality'] == 'audio_visual':
            # Process visual features: [B, 50, 7, 7] -> [B, 50, 49]
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)
            # Embed to audio dimension: [B, 50, 49] -> [B, 50, rnn_size]
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            # Fuse with transformer decoder
            fused_feat = self.transformer_decoder(x, vid_feat)
        else:
            fused_feat = x

        # === OUTPUT GENERATION ===
        
        # FFN layers
        for fnn_cnt in range(len(self.fnn_list) - 1):
            fused_feat = self.fnn_list[fnn_cnt](fused_feat)
        pred = self.fnn_list[-1](fused_feat)

        # Apply activations based on modality
        if self.params['modality'] == 'audio':
            if self.params['multiACCDOA']:
                # Audio-only: [B, 50, 117]
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 3, 13)
                doa_pred = self.doa_act(pred[:, :, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, :, 2:3, :])
                pred = torch.cat((doa_pred, dist_pred), dim=3)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
            else:
                # Audio-only single: [B, 50, 39]
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 13)
                doa_pred = self.doa_act(pred[:, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, 2:3, :])
                pred = torch.cat((doa_pred, dist_pred), dim=2)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
        else:  # audio_visual
            if self.params['multiACCDOA']:
                # Audio-visual: [B, 50, 156] - includes onscreen
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 4, 13)
                doa_pred = self.doa_act(pred[:, :, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, :, 2:3, :])
                onscreen_pred = self.onscreen_act(pred[:, :, :, 3:4, :])
                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=3)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
            else:
                # Audio-visual single: [B, 50, 52]
                pred = pred.reshape(pred.size(0), pred.size(1), 4, 13)
                doa_pred = self.doa_act(pred[:, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, 2:3, :])
                onscreen_pred = self.onscreen_act(pred[:, :, 3:4, :])
                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=2)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)

        return pred

    def calculate_flops(self, audio_input, video_input=None):
        """Calculate FLOPs for ASMA-SELD AudioVisual model."""
        total_flops = 0
        
        # Audio processing FLOPs
        # Conv layers with SE blocks
        conv_flops = 0
        input_h, input_w = 250, 64  # Initial audio dimensions
        input_channels = 4
        
        for i in range(self.params['nb_conv_blocks']):
            output_channels = self.params['nb_conv_filters']
            kernel_size = 3
            
            # Calculate output dimensions after pooling
            pool_h, pool_w = self.params['t_pool_size'][i], self.params['f_pool_size'][i]
            output_h = input_h // pool_h
            output_w = input_w // pool_w
            
            # Conv + SE block FLOPs
            conv_flops += output_h * output_w * kernel_size * kernel_size * input_channels * output_channels
            # SE block adds minimal overhead
            se_flops = output_channels * 2  # Squeeze + excitation approximation
            conv_flops += se_flops
            
            input_h, input_w = output_h, output_w
            input_channels = output_channels
        
        # RNN FLOPs
        rnn_hidden = self.params['rnn_size']
        seq_len = 50  # label_sequence_length
        rnn_flops = seq_len * rnn_hidden * rnn_hidden * 4 * self.params['nb_rnn_layers']
        
        # Self-attention FLOPs
        attn_flops = seq_len * seq_len * rnn_hidden * self.params['nb_self_attn_layers']
        
        # Transformer fusion FLOPs (for audio-visual)
        transformer_flops = 0
        if video_input is not None:
            # Video processing
            video_flops = 50 * 49 * 128  # Video feature processing
            # Cross-modal transformer
            transformer_flops = seq_len * seq_len * rnn_hidden * self.params['nb_transformer_layers'] * 2
            transformer_flops += video_flops
        
        total_flops = conv_flops + rnn_flops + attn_flops + transformer_flops
        return total_flops

    def get_model_stats(self):
        """Get comprehensive model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Create dummy inputs
        dummy_audio = torch.randn(1, 4, 250, 64)
        dummy_video = torch.randn(1, 50, 49)
        
        flops = self.calculate_flops(dummy_audio, dummy_video)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'estimated_flops': flops,
            'gflops': flops / 1e9,
            'model_name': 'ASMA-SLED'
        }


if __name__ == '__main__':
    # Test ASMA-SELD Audio-Visual
    params = {
        'nb_conv_blocks': 3,
        'nb_conv_filters': 64,
        'f_pool_size': [4, 4, 2],
        't_pool_size': [5, 1, 1],
        'dropout': 0.05,
        'rnn_size': 128,
        'nb_rnn_layers': 2,
        'nb_self_attn_layers': 2,
        'nb_attn_heads': 8,
        'nb_fnn_layers': 1,
        'fnn_size': 128,
        'nb_mels': 64,
        'max_polyphony': 3,
        'nb_classes': 13,
        'multiACCDOA': True,
        'modality': 'audio_visual',
        'resnet_feature_size': 49,
        'nb_transformer_layers': 2
    }

    model = ASMA_SELD_AudioVisual(params)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ASMA-SELD Audio-Visual created successfully!")
    print(f"Total Parameters: {total_params:,}")
    print(f"Target: ~2.5M parameters")
    
    # Test forward pass
    dummy_audio = torch.rand([2, 2, 251, 64])
    dummy_video = torch.rand([2, 50, 7, 7])
    output = model(dummy_audio, dummy_video)
    
    print(f"Forward pass successful!")
    print(f"Audio input shape: {dummy_audio.shape}")
    print(f"Video input shape: {dummy_video.shape}")
    print(f"Output shape: {output.shape} (expected: [2, 50, 156])")
    print(f"Ready for audio-visual training!")
