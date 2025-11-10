#!/usr/bin/env python3
"""
ASMA-SLED Ablation: No Transformer Fusion

Removes the transformer decoder for audio-visual fusion, using simple concatenation.
Expected DOA degradation: ~1-2° (from 22.5° to ~24°)

This variant tests the hypothesis that advanced cross-modal fusion contributes
moderately to spatial localization performance.

Author: ASMA-SLED Ablation Study
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
        x = self.se(x)  # V3's SE attention
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ASMA_SLED_NoTransformer(nn.Module):
    """
    ASMA-SLED without Transformer Fusion
    
    ABLATION: Removes transformer decoder, uses simple concatenation for audio-visual fusion.
    Expected impact: Moderate DOA degradation (~1-2°)
    
    Key differences from full ASMA-SLED:
    1. NO transformer decoder for cross-modal fusion
    2. Uses simple concatenation + linear projection instead
    3. All other components remain identical (phase encoding, SE blocks)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # === AUDIO PATH (unchanged) ===
        
        # Phase encoder for stereo spatial cues (KEPT)
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4)
        )
        
        # Conv layers with SE blocks (KEPT)
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(params['nb_conv_blocks']):
            in_channels = params['nb_conv_filters'] if conv_cnt > 0 else 6  # 2 stereo + 4 phase
            self.conv_blocks.append(SEConvBlock(
                in_channels=in_channels,
                out_channels=params['nb_conv_filters'],
                pool_size=(params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]),
                dropout=params['dropout']
            ))

        # GRU for temporal modeling (unchanged)
        self.gru_input_dim = params['nb_conv_filters'] * int(np.floor(params['nb_mels'] / np.prod(params['f_pool_size'])))
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=params['rnn_size'],
            num_layers=params['nb_rnn_layers'],
            batch_first=True,
            dropout=params['dropout'],
            bidirectional=True
        )

        # Self-attention layers (unchanged)
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

        # === SIMPLE FUSION PATH (replaces transformer) ===
        if params['modality'] == 'audio_visual':
            # Simple visual processing (instead of transformer)
            self.visual_embed_to_d_model = nn.Linear(
                in_features=params['resnet_feature_size'], 
                out_features=params['rnn_size']
            )
            
            # NO TRANSFORMER DECODER - This is the key ablation
            # Instead, use simple concatenation + projection
            self.fusion_projection = nn.Linear(
                in_features=params['rnn_size'] * 2,  # audio + video concatenated
                out_features=params['rnn_size']
            )

        # === OUTPUT PATH (unchanged) ===
        
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
        Forward pass WITHOUT transformer fusion.
        
        Args:
            audio_feat: [B, 2, 251, 64] - stereo spectrogram
            vid_feat: [B, 50, 7, 7] - visual feature map (optional)
            
        Returns:
            pred: Same as full ASMA-SLED but with simple fusion
        """
        # === AUDIO PROCESSING (unchanged) ===
        
        # Phase encoding (KEPT)
        phase_diff = audio_feat[:, 0:1, :, :] - audio_feat[:, 1:2, :, :]
        phase_feat = F.relu(self.phase_encoder(phase_diff))
        x = torch.cat([audio_feat, phase_feat], dim=1)  # [B, 6, T, F]

        # SE-enhanced convolution (KEPT)
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Reshape for temporal processing
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        # GRU temporal modeling (unchanged)
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]

        # Self-attention (unchanged)
        for mhsa, ln in zip(self.mhsa_layers, self.layer_norms):
            x_in = x
            x, _ = mhsa(x_in, x_in, x_in)
            x = x + x_in
            x = ln(x)

        # === SIMPLE FUSION (replaces transformer) ===
        
        if vid_feat is not None and self.params['modality'] == 'audio_visual':
            # Process visual features: [B, 50, 7, 7] -> [B, 50, 49]
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)
            # Embed to audio dimension: [B, 50, 49] -> [B, 50, rnn_size]
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            
            # ABLATION: Simple concatenation instead of transformer
            concatenated = torch.cat([x, vid_feat], dim=-1)  # [B, 50, 2*rnn_size]
            fused_feat = self.fusion_projection(concatenated)  # [B, 50, rnn_size]
        else:
            fused_feat = x

        # === OUTPUT GENERATION (unchanged) ===
        
        for fnn_cnt in range(len(self.fnn_list) - 1):
            fused_feat = self.fnn_list[fnn_cnt](fused_feat)
        pred = self.fnn_list[-1](fused_feat)

        # Apply activations (unchanged)
        if self.params['modality'] == 'audio':
            if self.params['multiACCDOA']:
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 3, 13)
                doa_pred = self.doa_act(pred[:, :, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, :, 2:3, :])
                pred = torch.cat((doa_pred, dist_pred), dim=3)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
            else:
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 13)
                doa_pred = self.doa_act(pred[:, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, 2:3, :])
                pred = torch.cat((doa_pred, dist_pred), dim=2)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
        else:  # audio_visual
            if self.params['multiACCDOA']:
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 4, 13)
                doa_pred = self.doa_act(pred[:, :, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, :, 2:3, :])
                onscreen_pred = self.onscreen_act(pred[:, :, :, 3:4, :])
                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=3)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
            else:
                pred = pred.reshape(pred.size(0), pred.size(1), 4, 13)
                doa_pred = self.doa_act(pred[:, :, 0:2, :])
                dist_pred = self.dist_act(pred[:, :, 2:3, :])
                onscreen_pred = self.onscreen_act(pred[:, :, 3:4, :])
                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=2)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)

        return pred


# Alias for compatibility with training scripts
SELDNet_NoTransformer = ASMA_SLED_NoTransformer


if __name__ == '__main__':
    # Test the no-transformer ablation model
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

    model = ASMA_SLED_NoTransformer(params)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ASMA-SLED No Transformer Ablation created successfully!")
    print(f"Total Parameters: {total_params:,}")
    print(f"Expected DOA degradation: ~1-2° (from 22.5° to ~24°)")
    
    # Test forward pass
    dummy_audio = torch.rand([2, 2, 251, 64])
    dummy_video = torch.rand([2, 50, 7, 7])
    output = model(dummy_audio, dummy_video)
    
    print(f"Forward pass successful!")
    print(f"Audio input shape: {dummy_audio.shape}")
    print(f"Video input shape: {dummy_video.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Ablation: Transformer fusion DISABLED (simple concatenation used)")