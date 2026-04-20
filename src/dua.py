"""
DUA: Distribution Uncertainty Adaptation (Mirza et al., 2022).
https://arxiv.org/abs/2202.11763

Adapts BatchNorm running statistics (mu, sigma^2) by forwarding unlabeled
test images one at a time, each expanded into N augmented copies, with
geometrically decaying momentum. No gradients or optimizer required.

Paper algorithm (from methods/dua.py in the official repo):
  For each test image i:
    1. Create 64 augmented copies of the image (augmentation batch)
    2. mom_new = mom_pre * decay_factor
    3. For each BN layer: m.train(); m.momentum = mom_new + min_momentum_constant
    4. Forward the 64-copy batch (no_grad)
    5. For each BN layer: m.eval()
    6. mom_pre = mom_new  (decayed value carries forward)

Two details that differ from naive interpretations:
  - Processing unit is a single image, not a multi-image batch
  - The momentum floor is ADDITIVE: mom_new + min_momentum_constant,
    not max(min_momentum, ...). At low momentum this gives a slightly
    higher effective momentum than a hard clip.

For detection models, rotation augmentation (used in the paper's SSH
variant) would break spatial structure. We use photometric augmentation
(brightness, contrast, noise) + horizontal flip instead.

Unlike TENT, DUA:
  - Requires zero memory for gradients or optimizer state
  - Is stable even with batch size 1
  - Cannot overfit BN affine params (gamma/beta are frozen)
  - Needs enough test images for stats to converge (~100+)
"""
import copy

import torch
import torch.nn as nn
from ultralytics import YOLO


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Photometric augmentation + random horizontal flip for a single image.

    Preserves spatial structure so augmented views stay valid detection inputs.
    Applied independently to each of the n_augments copies in DUA.

    Args:
        x: single image tensor [3, H, W], float32 in [0, 1]
    Returns:
        augmented tensor, same shape
    """
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        x = x.flip(-1)

    # Brightness shift
    shift = torch.empty(1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    x = (x + shift).clamp(0, 1)

    # Contrast scaling
    scale = torch.empty(1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    mean = x.mean(dim=(1, 2), keepdim=True)
    x = ((x - mean) * scale + mean).clamp(0, 1)

    # Gaussian noise
    x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)

    return x


class DUA:
    def __init__(
        self,
        model: YOLO,
        decay_factor: float = 0.94,
        min_momentum_constant: float = 0.005,
        mom_pre: float = 0.1,
        n_augments: int = 64,
    ):
        """
        Args:
            model: ultralytics YOLO wrapper (model.model is the nn.Module)
            decay_factor: multiplicative decay per adapt() call (paper default: 0.94)
            min_momentum_constant: additive floor on BN momentum (paper default: 0.005)
                Added to decayed momentum: effective_mom = mom_new + min_momentum_constant
            mom_pre: initial momentum before any decay (paper default: 0.1)
            n_augments: augmented copies per image for BN stat estimation
                (paper default: 64; reduce if VRAM is tight)
        """
        self.model = model
        self.decay_factor = decay_factor
        self.min_momentum = min_momentum_constant
        self.n_augments = n_augments
        self._mom = mom_pre
        self._initial_mom = mom_pre
        self._original_state = copy.deepcopy(model.model.state_dict())

        # Freeze all parameters — DUA is gradient-free
        self.model.model.eval()
        for p in self.model.model.parameters():
            p.requires_grad_(False)

        self._bn_layers = [
            m for m in self.model.model.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
        ]
        print(f"[DUA] Adapting running stats of {len(self._bn_layers)} BN layers  "
              f"n_augments={n_augments}  decay={decay_factor}  min_mom={min_momentum_constant}")

    def adapt(self, x: torch.Tensor) -> None:
        """Update BN running statistics for a single image with decaying momentum.

        Mirrors the paper's per-image loop exactly: for each call, the single
        image is expanded into n_augments augmented copies, forwarded once, and
        the momentum is decayed for the next call.

        Args:
            x: single image tensor [3, H, W] or [1, 3, H, W], on device, float32 in [0, 1]

        Call this once per test image in order. After all images the running
        stats reflect the target distribution. Then call model.val() normally.
        """
        if x.ndim == 4:
            x = x.squeeze(0)  # [1, 3, H, W] → [3, H, W]

        # Decay momentum — paper formula: mom_new = mom_pre * decay_factor
        # Effective BN momentum = mom_new + min_momentum_constant (additive floor)
        mom_new = self._mom * self.decay_factor
        self._mom = mom_new
        effective_mom = mom_new + self.min_momentum

        # Build augmented batch: n_augments independently augmented copies
        augmented = torch.stack([_augment(x) for _ in range(self.n_augments)])

        # Switch BN to train mode so PyTorch updates running stats
        for bn in self._bn_layers:
            bn.train()
            bn.momentum = effective_mom

        with torch.no_grad():
            self.model.model(augmented)

        # Restore eval mode — running stats are now updated
        for bn in self._bn_layers:
            bn.eval()

    def reset(self) -> None:
        """Restore source-domain weights and running stats."""
        self.model.model.load_state_dict(self._original_state)
        self.model.model.eval()
        for p in self.model.model.parameters():
            p.requires_grad_(False)
        self._mom = self._initial_mom
