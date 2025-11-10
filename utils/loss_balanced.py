"""
Balanced Loss Function for SELD

Addresses the audio-visual loss discrepancy by:
1. Properly weighting MSE and BCE losses
2. Normalizing loss scales
3. Separate distance loss weighting

Author: Enhanced version
Date: November 2025
"""

import torch
import torch.nn as nn


class SELDLossADPITBalanced(nn.Module):
    """
    Balanced ADPIT loss with proper weighting for audio-visual training
    
    Key improvements:
    - Weighted combination of MSE and BCE losses
    - Separate distance loss term
    - Normalized loss scales
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.modality = params['modality']
        
        # Loss weights (tunable)
        self.accdoa_weight = params.get('loss_accdoa_weight', 1.0)  # For x, y prediction
        self.dist_weight = params.get('loss_dist_weight', 1.0)  # For distance prediction
        self.onscreen_weight = params.get('loss_onscreen_weight', 0.5)  # For onscreen prediction (reduced)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def _mse(self, output, target):
        return self.mse_loss(output, target).mean(dim=2)  # class-wise frame-level

    def _bce(self, output, target):
        return self.bce_loss(output, target).mean(dim=2)  # class-wise frame-level

    def _make_target_perms(self, target_A0, target_B0, target_B1, target_C0, target_C1, target_C2):
        """Make 13 possible target permutations"""
        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1

        list_target_perms = [target_A0A0A0 + pad4A,
                             target_B0B0B1 + pad4B, target_B0B1B0 + pad4B, target_B0B1B1 + pad4B,
                             target_B1B0B0 + pad4B, target_B1B0B1 + pad4B, target_B1B1B0 + pad4B,
                             target_C0C1C2 + pad4C, target_C0C2C1 + pad4C, target_C1C0C2 + pad4C,
                             target_C1C2C0 + pad4C, target_C2C0C1 + pad4C, target_C2C1C0 + pad4C]
        return list_target_perms

    def forward(self, output, target):
        """
        Balanced ADPIT loss computation
        
        Args:
            output:
                audio:        (batch_size, 50, 117) -> 3 tracks x (x, y, dist) x 13 classes
                audio_visual: (batch_size, 50, 156) -> 3 tracks x (x, y, dist, onscreen) x 13 classes
            target:
                audio:        (batch_size, 50, 6, 4, 13) -> 6 tracks x (sed, x, y, dist) x 13 classes
                audio_visual: (batch_size, 50, 6, 5, 13) -> 6 tracks x (sed, x, y, dist, onscreen) x 13 classes
        """
        num_bs = target.shape[0]
        num_frame = target.shape[1]
        num_track = 3
        num_element = target.shape[3] - 1
        num_class = target.shape[4]
        num_permutation = 13

        # Reshape output
        output_reshaped = output.reshape(num_bs, num_frame, num_track, num_element, num_class)
        
        # Split into components
        output_doa = output_reshaped[:, :, :, 0:2, :]  # x, y (DOA)
        output_dist = output_reshaped[:, :, :, 2:3, :]  # distance
        
        # Combine DOA for ACCDOA loss (act*x, act*y)
        output_accdoa = output_doa.reshape(num_bs, num_frame, -1, num_class)
        
        # Extract target permutations for DOA
        target_doa_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:3, :]  # act * (x, y)
        target_doa_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:3, :]
        target_doa_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:3, :]
        target_doa_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:3, :]
        target_doa_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:3, :]
        target_doa_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:3, :]
        
        list_target_doa_perms = self._make_target_perms(
            target_doa_A0, target_doa_B0, target_doa_B1,
            target_doa_C0, target_doa_C1, target_doa_C2
        )
        
        # Compute DOA loss for each permutation
        list_loss_doa = []
        for each_target_doa in list_target_doa_perms:
            list_loss_doa.append(self._mse(output_accdoa, each_target_doa))
        
        # Extract target permutations for distance
        output_dist_flat = output_dist.reshape(num_bs, num_frame, -1, num_class)
        target_dist_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 3:4, :]  # act * dist
        target_dist_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 3:4, :]
        target_dist_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 3:4, :]
        target_dist_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 3:4, :]
        target_dist_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 3:4, :]
        target_dist_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 3:4, :]
        
        list_target_dist_perms = self._make_target_perms(
            target_dist_A0, target_dist_B0, target_dist_B1,
            target_dist_C0, target_dist_C1, target_dist_C2
        )
        
        # Compute distance loss for each permutation
        list_loss_dist = []
        for each_target_dist in list_target_dist_perms:
            list_loss_dist.append(self._mse(output_dist_flat, each_target_dist))
        
        # Handle onscreen prediction for audio-visual
        if self.modality == 'audio':
            list_loss_onscreen = [0.0] * num_permutation
        else:
            output_onscreen = output_reshaped[:, :, :, 3:4, :]
            output_onscreen = output_onscreen.reshape(num_bs, num_frame, -1, num_class)
            
            target_onscreen_A0 = target[:, :, 0, 4:5, :]
            target_onscreen_B0 = target[:, :, 1, 4:5, :]
            target_onscreen_B1 = target[:, :, 2, 4:5, :]
            target_onscreen_C0 = target[:, :, 3, 4:5, :]
            target_onscreen_C1 = target[:, :, 4, 4:5, :]
            target_onscreen_C2 = target[:, :, 5, 4:5, :]
            
            list_target_onscreen_perms = self._make_target_perms(
                target_onscreen_A0, target_onscreen_B0, target_onscreen_B1,
                target_onscreen_C0, target_onscreen_C1, target_onscreen_C2
            )
            
            list_loss_onscreen = []
            for each_target_onscreen in list_target_onscreen_perms:
                list_loss_onscreen.append(self._bce(output_onscreen, each_target_onscreen))
        
        # Choose permutation based on DOA loss (primary metric)
        loss_doa_min = torch.min(torch.stack(list_loss_doa, dim=0), dim=0).indices
        
        # Compute weighted combined loss using the chosen permutation
        loss_sum = 0
        for i in range(num_permutation):
            # Weighted combination: DOA + Distance + Onscreen
            weighted_loss = (
                self.accdoa_weight * list_loss_doa[i] +
                self.dist_weight * list_loss_dist[i] +
                self.onscreen_weight * list_loss_onscreen[i]
            )
            loss_sum += weighted_loss * (loss_doa_min == i)
        
        loss = loss_sum.mean()
        return loss


if __name__ == '__main__':
    # Test loss function
    params = {
        'modality': 'audio_visual',
        'loss_accdoa_weight': 1.0,
        'loss_dist_weight': 1.0,
        'loss_onscreen_weight': 0.5
    }
    
    loss_fn = SELDLossADPITBalanced(params)
    
    # Test data
    batch_size, time_steps, num_classes = 4, 50, 13
    output = torch.randn(batch_size, time_steps, 156)  # 3 tracks x 4 elements x 13 classes
    target = torch.randn(batch_size, time_steps, 6, 5, num_classes)
    
    loss = loss_fn(output, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test audio-only
    params['modality'] = 'audio'
    loss_fn_audio = SELDLossADPITBalanced(params)
    output_audio = torch.randn(batch_size, time_steps, 117)  # 3 tracks x 3 elements x 13 classes
    target_audio = torch.randn(batch_size, time_steps, 6, 4, num_classes)
    
    loss_audio = loss_fn_audio(output_audio, target_audio)
    print(f"Audio loss: {loss_audio.item():.4f}")
