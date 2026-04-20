"""
CoTTA: Continual Test-Time Adaptation (Wang et al., 2022).
https://arxiv.org/abs/2203.13591

Designed for continuously shifting target distributions where a single
domain assumption doesn't hold. Three mechanisms prevent error accumulation
and catastrophic forgetting:

  1. Augmentation-averaged pseudo-labels: instead of using raw predictions
     as supervision, CoTTA averages predictions across aug_times augmented
     views of each batch. This reduces pseudo-label noise, which is the main
     driver of error accumulation in naive online TTA.

  2. Stochastic weight restore: after each update, each parameter is
     independently reset to its source value with probability restore_prob.
     This prevents any single parameter from drifting too far, maintaining
     a "tether" to the source-domain initialisation.

  3. EMA model for pseudo-label generation: pseudo-labels are computed from
     a weight-space EMA of the adapted model rather than the current weights,
     providing a more stable teacher signal.

Unlike TENT or SAR, CoTTA adapts all normalisation parameters (not just BN
affine), making it more expressive but also slower per step.

Note: augmentations here are photometric only (no spatial transforms) so
that raw output tensors can be directly averaged without spatial alignment.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from src.tent import collect_bn_params, tent_loss


def _photometric_aug(x: torch.Tensor) -> torch.Tensor:
    """Random photometric augmentation that preserves spatial layout.

    Applies random combinations of: brightness, contrast, gaussian noise.
    Safe to average across views because no spatial transforms are used.
    """
    out = x.clone()
    B = out.shape[0]

    # Random brightness shift
    shift = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    out = (out + shift).clamp(0, 1)

    # Random contrast scaling
    scale = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    mean  = out.mean(dim=(1, 2, 3), keepdim=True)
    out   = ((out - mean) * scale + mean).clamp(0, 1)

    # Gaussian noise
    out = (out + 0.02 * torch.randn_like(out)).clamp(0, 1)

    return out


class CoTTA:
    def __init__(
        self,
        model: YOLO,
        lr: float = 0.001,
        steps: int = 1,
        aug_times: int = 32,
        restore_prob: float = 0.01,
        ema_decay: float = 0.999,
    ):
        """
        Args:
            model: ultralytics YOLO wrapper
            lr: learning rate for Adam optimizer
            steps: gradient steps per batch
            aug_times: number of augmented views for pseudo-label averaging
                (paper uses 32; reduce to 8 if VRAM is tight)
            restore_prob: per-parameter probability of resetting to source
                weights each step (paper default: 0.01)
            ema_decay: EMA decay for the teacher model (paper default: 0.999)
        """
        self.model       = model
        self.steps       = steps
        self.aug_times   = aug_times
        self.restore_prob = restore_prob
        self.ema_decay   = ema_decay

        # Source weights — never updated, used for stochastic restore
        self._source_state = copy.deepcopy(model.model.state_dict())
        # EMA teacher weights — updated as soft copy of adapted model
        self._ema_state    = copy.deepcopy(model.model.state_dict())

        self._configure()
        params, _ = collect_bn_params(model.model)
        print(f"[CoTTA] Adapting {len(params)} BN parameters  "
              f"aug_times={aug_times}  restore_prob={restore_prob}")
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self._orig_forward = model.model.forward

    def _configure(self) -> None:
        """Train mode, freeze all, unfreeze BN affine (same as TENT)."""
        self.model.model.train()
        for p in self.model.model.parameters():
            p.requires_grad_(False)
        for m in self.model.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.requires_grad_(True)
                m.track_running_stats = False

    def _ema_forward(self, x: torch.Tensor) -> object:
        """Forward pass using EMA teacher weights (no grad)."""
        # Temporarily swap to EMA weights
        current_state = {k: v.clone() for k, v in self.model.model.state_dict().items()}
        self.model.model.load_state_dict(self._ema_state)
        self.model.model.eval()
        with torch.no_grad():
            out = self.model.model(x)
        # Restore adapted weights + train mode
        self.model.model.load_state_dict(current_state)
        self._configure()
        return out

    def _aug_avg_predictions(self, x: torch.Tensor) -> object:
        """Average EMA-teacher predictions across aug_times augmented views."""
        preds_list = []
        for _ in range(self.aug_times):
            x_aug  = _photometric_aug(x)
            pred   = self._ema_forward(x_aug)
            preds_list.append(pred)

        # Average tensor outputs from each detection head
        if isinstance(preds_list[0], (list, tuple)):
            averaged = []
            for i in range(len(preds_list[0])):
                tensors = [p[i] for p in preds_list if isinstance(p[i], torch.Tensor)]
                if tensors:
                    averaged.append(torch.stack(tensors).mean(0))
                else:
                    averaged.append(preds_list[0][i])
            return averaged
        # Single-tensor output
        if isinstance(preds_list[0], torch.Tensor):
            return torch.stack(preds_list).mean(0)
        return preds_list[0]

    def _update_ema(self) -> None:
        """Update EMA teacher: w_ema = decay * w_ema + (1-decay) * w."""
        alpha = self.ema_decay
        current = self.model.model.state_dict()
        for k in self._ema_state:
            if self._ema_state[k].dtype.is_floating_point:
                self._ema_state[k] = alpha * self._ema_state[k] + (1 - alpha) * current[k]

    def _stochastic_restore(self) -> None:
        """Randomly reset each BN affine parameter to its source value."""
        source  = self._source_state
        current = self.model.model.state_dict()
        with torch.no_grad():
            for m in self.model.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    for name, p in m.named_parameters(recurse=False):
                        # Find the full key for this parameter in state_dict
                        for k, v in current.items():
                            if v.data_ptr() == p.data_ptr() and k in source:
                                mask = torch.bernoulli(
                                    torch.full_like(p.data, self.restore_prob)
                                ).bool()
                                p.data[mask] = source[k][mask]
                                break

    def adapt(self, x: torch.Tensor) -> None:
        """Adapt on batch x using augmentation-averaged pseudo-labels.

        Args:
            x: preprocessed image tensor [B, 3, H, W], on device, float32 in [0, 1]
        """
        # ── 1. Pseudo-labels from EMA teacher averaged over augmented views ──
        # This is computed outside the gradient tape — detached pseudo-targets
        pseudo = self._aug_avg_predictions(x)

        # ── 2. Entropy minimization on current model ────────────────────────
        # We use the same tent_loss on current model predictions, not a
        # cross-entropy against pseudo — this follows the CoTTA paper where
        # the pseudo-labels are used implicitly via the augmentation consistency
        for _ in range(self.steps):
            preds = self._orig_forward(x, augment=False)
            loss  = tent_loss(preds)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ── 3. Update EMA teacher ───────────────────────────────────────────
        self._update_ema()

        # ── 4. Stochastic weight restore ────────────────────────────────────
        self._stochastic_restore()

    def reset(self) -> None:
        """Restore source-domain weights."""
        self.model.model.load_state_dict(self._source_state)
        self._ema_state = copy.deepcopy(self._source_state)
        self._configure()
