"""Gram Anchoring Loss for structural similarity preservation.

Encourages the student to preserve the pairwise patch-similarity structure
(Gram matrix) of a "Gram teacher" -- typically an earlier checkpoint or a
periodically updated copy. This acts as a regularizer that prevents
catastrophic forgetting of learned spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """Gram matrix anchoring loss between student and teacher features.

    Computes per-image Gram matrices from L2-normalized patch features and
    minimizes the MSE between student and teacher Gram matrices.

    Args:
        remove_neg: If True, clamp Gram matrix entries to non-negative
            values before computing the loss. This focuses the loss on
            positive correlations and ignores anti-correlated patches.
    """

    def __init__(self, remove_neg: bool = True):
        super().__init__()
        self.remove_neg = remove_neg

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gram anchoring loss.

        Operates per-image: computes separate Gram matrices for each image's
        patch features, then averages the MSE across the batch.

        Args:
            student_features: [B, N, D] student patch features.
            teacher_features: [B, N, D] teacher (Gram anchor) patch features.

        Returns:
            Scalar MSE loss between student and teacher Gram matrices.
        """
        B = student_features.shape[0]

        # L2 normalize along feature dimension
        student_norm = F.normalize(student_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)

        # Vectorized Gram matrices: [B, N, N]
        student_gram = torch.bmm(student_norm, student_norm.transpose(1, 2))
        teacher_gram = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2))

        if self.remove_neg:
            student_gram = torch.clamp(student_gram, min=0.0)
            teacher_gram = torch.clamp(teacher_gram, min=0.0)

        return F.mse_loss(student_gram, teacher_gram)
