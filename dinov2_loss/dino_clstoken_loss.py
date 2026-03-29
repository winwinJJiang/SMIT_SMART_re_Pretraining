"""DINO CLS token self-distillation loss with Sinkhorn-Knopp centering.

The student learns to match the teacher's CLS token distribution via
cross-entropy on soft targets produced by Sinkhorn-Knopp normalization.
Same-crop pairs between global views are excluded to avoid trivial solutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DINOLoss(nn.Module):
    """DINO self-distillation loss on CLS tokens.

    Args:
        out_dim: Output projection dimension (vocabulary size).
        student_temp: Temperature for student softmax.
        center_momentum: EMA momentum for teacher centering.
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

        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def sinkhorn_knopp(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """Apply Sinkhorn-Knopp normalization to teacher outputs.

        Produces a balanced soft assignment by alternating row and column
        normalization of the exponentiated similarity matrix.

        Args:
            teacher_output: [B, K] teacher logits (already centered).
            teacher_temp: Temperature for teacher sharpening.
            n_iterations: Number of Sinkhorn-Knopp iterations.

        Returns:
            Normalized teacher probabilities [B, K].
        """
        Q = torch.exp(teacher_output / teacher_temp).T  # [K, B]

        B = Q.shape[1]
        K = Q.shape[0]

        # Distributed: gather total batch size for correct normalization
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_B = B * world_size

        Q /= Q.sum()  # normalize full matrix

        for _ in range(n_iterations):
            # Row normalization: each prototype gets equal mass
            row_sum = Q.sum(dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(row_sum)
            Q /= row_sum
            Q /= K

            # Column normalization: each sample gets equal mass (local B)
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.T  # [B, K]

    def forward(
        self,
        student_output_list: list[torch.Tensor],
        teacher_output: torch.Tensor,
        teacher_temp: float,
    ) -> torch.Tensor:
        """Compute DINO CLS token loss.

        Args:
            student_output_list: List of student CLS outputs, each [B, out_dim].
                First n_teacher_crops entries are global crops; rest are local.
            teacher_output: [n_teacher_crops, B, out_dim] teacher CLS outputs.
            teacher_temp: Current teacher temperature (annealed during training).

        Returns:
            Scalar loss averaged over all valid crop pairs.
        """
        n_teacher_crops = teacher_output.shape[0]

        # Student log-probabilities
        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_output_list
        ]

        # Teacher probabilities via centering + Sinkhorn-Knopp
        teacher_centered = teacher_output - self.center  # [T, B, out_dim]
        teacher_probs = [
            self.sinkhorn_knopp(teacher_centered[t], teacher_temp)
            for t in range(n_teacher_crops)
        ]

        # Cross-entropy across all (teacher, student) pairs, skipping same-crop
        total_loss = torch.tensor(0.0, device=teacher_output.device)
        n_pairs = 0

        for t_idx in range(n_teacher_crops):
            for s_idx in range(len(student_output_list)):
                # Skip same global crop (diagonal for global-global pairs)
                if s_idx == t_idx:
                    continue

                loss = -torch.sum(
                    teacher_probs[t_idx] * student_log_probs[s_idx],
                    dim=-1,
                )
                total_loss = total_loss + loss.mean()
                n_pairs += 1

        total_loss = total_loss / max(n_pairs, 1)

        # Update center with current teacher batch
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Update teacher center via exponential moving average.

        Args:
            teacher_output: [n_teacher_crops, B, out_dim] teacher CLS outputs.
        """
        # Average over crops and batch: [out_dim]
        batch_center = teacher_output.mean(dim=(0, 1), keepdim=False)

        # Distributed: average across GPUs
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()

        # EMA update (in-place to preserve register_buffer)
        self.center.mul_(self.center_momentum).add_(
            batch_center.unsqueeze(0), alpha=1.0 - self.center_momentum
        )
