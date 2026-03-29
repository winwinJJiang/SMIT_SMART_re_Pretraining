"""KoLeo entropy regularization loss.

Implements the Kozachenko-Leonenko (KoLeo) differential entropy estimator
as a regularizer that encourages uniform feature distribution on the unit
hypersphere. This prevents representation collapse by penalizing features
that cluster too tightly.

Reference: Sablayrolles et al., "Spreading vectors for similarity search", ICLR 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class KoLeoLoss(nn.Module):
    """KoLeo entropy regularizer for uniform feature distribution.

    Maximizes the Kozachenko-Leonenko entropy estimate by encouraging each
    feature vector to be far from its nearest neighbor on the unit sphere.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute KoLeo regularization loss.

        Args:
            features: [B, D] feature vectors (will be L2-normalized).

        Returns:
            Scalar loss: negative log of mean nearest-neighbor distance.
            Lower (more negative) values indicate more uniform distributions.
        """
        # Fix #10: guard for degenerate cases
        if features.shape[0] < 2:
            return features.sum() * 0.0  # Fix: preserve autograd path

        # Gather features from all GPUs for accurate NN computation
        feats = F.normalize(features, p=2, dim=-1)
        if dist.is_initialized():
            feats = self._all_gather(feats)

        # Pairwise cosine similarity (features already normalized)
        dots = feats @ feats.T  # [N, N]

        # Exclude self-similarity by masking diagonal (non-in-place for autograd safety)
        n = dots.shape[0]
        eye_mask = torch.eye(n, device=dots.device, dtype=torch.bool)
        dots = dots.masked_fill(eye_mask, float("-inf"))  # Fix #9: -inf instead of -1.0

        # Find nearest neighbor for each feature
        nn_indices = dots.argmax(dim=1)  # [N]

        # L2 distance to nearest neighbor
        nn_feats = feats[nn_indices]  # [N, D]
        dists = torch.norm(feats - nn_feats, p=2, dim=-1)  # [N]

        # KoLeo entropy estimate: -mean(log(dist))
        loss = -torch.log(dists + 1e-8).mean()

        return loss

    @staticmethod
    def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all GPUs and concatenate.

        Uses all_gather into a pre-allocated list to avoid in-place issues
        with autograd.

        Args:
            tensor: Local tensor [B, D].

        Returns:
            Concatenated tensor [B * world_size, D].
        """
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)

        # Replace current rank's copy with the original (preserves gradients)
        rank = dist.get_rank()
        gathered[rank] = tensor

        return torch.cat(gathered, dim=0)
