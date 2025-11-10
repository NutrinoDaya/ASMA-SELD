#!/usr/bin/env python3
"""
ASMA-SLED Ablation: No SE Blocks

Removes the Squeeze-and-Excitation attention blocks to test their impact.
Expected DOA degradation: ~1-2° (from 22.5° to ~24°)

This variant tests the hypothesis that channel attention contributes
moderately to spatial localization performance.

Author: ASMA-SLED Ablation Study
Date: November 9, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicConvBlock(nn.Module):
    """
    Basic convolutional block WITHOUT SE attention.
    Replaces SEConvBlock for ablation study.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=(5, 4), dropout=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # NO SE BLOCK - This is the key ablation
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # NO SE attention applied here
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ASMA_SLED_NoSE(nn.Module):
    """
    ASMA-SLED without SE (Squeeze-and-Excitation) Blocks
    
    ABLATION: Removes channel attention to test its contribution.
    Expected impact: Moderate DOA degradation (~1-2°)
    
    Key differences from full ASMA-SLED:
    1. Uses BasicConvBlock instead of SEConvBlock
    2. NO channel attention mechanism
    3. All other components remain identical (phase encoding, transformer)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # === AUDIO PATH (with phase encoding, without SE blocks) ===
        
        # Phase encoder for stereo spatial cues (KEPT)
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4)
        )
        
        # Conv layers WITHOUT SE blocks (key ablation)
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(params['nb_conv_blocks']):
            in_channels = params['nb_conv_filters'] if conv_cnt > 0 else 6  # 2 stereo + 4 phase
            self.conv_blocks.append(BasicConvBlock(  # Using BasicConvBlock instead of SEConvBlock
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

        # === VISUAL FUSION PATH (unchanged) ===
        if params['modality'] == 'audio_visual':
            self.visual_embed_to_d_model = nn.Linear(
                in_features=params['resnet_feature_size'], 
                out_features=params['rnn_size']
            )
            
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(
                d_model=params['rnn_size'],
                nhead=params['nb_attn_heads'],
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(
                self.transformer_decoder_layer,
                num_layers=params.get('nb_transformer_layers', 2)
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
        Forward pass WITHOUT SE attention blocks.
        
        Args:
            audio_feat: [B, 2, 251, 64] - stereo spectrogram
            vid_feat: [B, 50, 7, 7] - visual feature map (optional)
            
        Returns:
            pred: Same as full ASMA-SLED but without channel attention
        """
        # === AUDIO PROCESSING (with phase, without SE) ===
        
        # Phase encoding (KEPT)
        phase_diff = audio_feat[:, 0:1, :, :] - audio_feat[:, 1:2, :, :]
        phase_feat = F.relu(self.phase_encoder(phase_diff))
        x = torch.cat([audio_feat, phase_feat], dim=1)  # [B, 6, T, F]

        # Basic convolution WITHOUT SE attention (key ablation)
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

        # === VISUAL FUSION (unchanged) ===
        
        if vid_feat is not None and self.params['modality'] == 'audio_visual':
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            fused_feat = self.transformer_decoder(x, vid_feat)
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
SELDNet_NoSE = ASMA_SLED_NoSE


if __name__ == '__main__':
    # Test the no-SE ablation model
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

    model = ASMA_SLED_NoSE(params)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ASMA-SLED No SE Blocks Ablation created successfully!")
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
    print(f"Ablation: SE Blocks DISABLED (basic conv blocks used)")