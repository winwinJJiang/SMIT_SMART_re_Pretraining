"""iBOT masked patch prediction loss.

The student predicts teacher patch-level representations for masked positions
only, using Sinkhorn-Knopp normalized teacher targets. This encourages the
model to learn local spatial semantics complementary to the global CLS loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class iBOTPatchLoss(nn.Module):
    """iBOT patch-level self-distillation loss on masked tokens.

    Args:
        out_dim: Output projection dimension (vocabulary size).
        student_temp: Temperature for student softmax.
        center_momentum: EMA momentum for patch-level centering.
    """

    def __init__(
        self,
        out_dim: int = 65536,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Patch-level center: [1, 1, out_dim] to broadcast over [B, N, D]
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def sinkhorn_knopp_masked(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_masked_patches: int,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """Sinkhorn-Knopp normalization for masked patch tokens.

        Unlike the CLS version, the number of tokens varies per sample due
        to masking. We operate on the flattened set of masked patches.

        Args:
            teacher_output: [M, K] flattened masked teacher patch logits
                (already centered), where M = total masked patches in batch.
            teacher_temp: Temperature for teacher sharpening.
            n_masked_patches: Total masked patches across all GPUs.
            n_iterations: Number of Sinkhorn-Knopp iterations.

        Returns:
            Normalized teacher probabilities [M, K].
        """
        Q = torch.exp(teacher_output / teacher_temp).T  # [K, M]

        M = Q.shape[1]
        K = Q.shape[0]

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_M = n_masked_patches

        Q /= Q.sum()  # normalize full matrix

        for _ in range(n_iterations):
            # Row normalization: each prototype gets equal mass
            row_sum = Q.sum(dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(row_sum)
            Q /= row_sum
            Q /= K

            # Column normalization: each masked patch gets equal mass (local M)
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= M

        Q *= M
        return Q.T  # [M, K]

    def forward(
        self,
        student_patch_tokens: torch.Tensor,
        teacher_patch_tokens: torch.Tensor,
        student_masks_bool: torch.Tensor,
        teacher_temp: float,
    ) -> torch.Tensor:
        """Compute iBOT masked patch prediction loss.

        Args:
            student_patch_tokens: [B, N, out_dim] student patch-level outputs.
            teacher_patch_tokens: [B, N, out_dim] teacher patch-level outputs.
            student_masks_bool: [B, N] boolean mask where True = masked position.
            teacher_temp: Current teacher temperature.

        Returns:
            Scalar loss averaged over all masked patches.
        """
        # Extract masked positions
        masked_student = student_patch_tokens[student_masks_bool]  # [M, D]
        masked_teacher = teacher_patch_tokens[student_masks_bool]  # [M, D]

        n_masked = masked_student.shape[0]
        if n_masked == 0:
            return torch.tensor(0.0, device=student_patch_tokens.device)

        # Gather total masked count for balanced Sinkhorn-Knopp
        total_masked = torch.tensor(
            [n_masked], device=student_patch_tokens.device, dtype=torch.long
        )
        if dist.is_initialized():
            dist.all_reduce(total_masked)

        # Center teacher and apply Sinkhorn-Knopp
        masked_teacher_centered = masked_teacher - self.center.squeeze(0)
        teacher_probs = self.sinkhorn_knopp_masked(
            masked_teacher_centered,
            teacher_temp,
            n_masked_patches=total_masked.item(),
        )

        # Student log-probabilities
        student_log_probs = F.log_softmax(
            masked_student / self.student_temp, dim=-1
        )

        # Cross-entropy loss
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1)
        loss = loss.mean()

        # Update center
        self.update_center(teacher_patch_tokens, student_masks_bool)

        return loss

    @torch.no_grad()
    def update_center(
        self,
        teacher_patch_tokens: torch.Tensor,
        masks_bool: torch.Tensor,
    ) -> None:
        """Update patch-level center via EMA over all (not just masked) patches.

        Using all patches for center estimation provides a more stable and
        representative estimate of the feature distribution.

        Args:
            teacher_patch_tokens: [B, N, out_dim] teacher patch outputs.
            masks_bool: [B, N] boolean mask (unused here; center uses all patches).
        """
        # Mean over batch and spatial dims: [out_dim]
        batch_center = teacher_patch_tokens.mean(dim=(0, 1))

        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()

        # In-place update to preserve register_buffer
        self.center.mul_(self.center_momentum).add_(
            batch_center.view(1, 1, -1), alpha=1.0 - self.center_momentum
        )
