"""
TENT: Test-Time Adaptation via Entropy Minimization for YOLOv8.

Reference: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization" (ICLR 2021)
           https://github.com/DequanWang/tent

Key idea: at test time, update only BatchNorm affine parameters (gamma, beta)
by minimising prediction entropy — no labelled data needed.
"""

import copy
import torch
import torch.nn as nn
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def configure_model(model: nn.Module) -> nn.Module:
    """
    Prepare the model for TENT adaptation:
      - Set the whole model to eval (disables Dropout, freezes running stats etc.)
      - Re-enable BN layers in train mode so affine params get updated
      - Freeze everything except BN affine parameters
    """
    model.train()
    # First freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Then selectively unfreeze BN affine params
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            # Don't track running stats — use batch stats for adaptation
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model


def collect_bn_params(model: nn.Module):
    """Return (params, param_names) for only the BN affine parameters."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for pnm, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pnm}")
    return params, names


# ---------------------------------------------------------------------------
# Entropy loss on detection class scores
# ---------------------------------------------------------------------------

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of a softmax distribution.
    logits: (..., num_classes)  — raw class scores from YOLOv8 detection head
    """
    probs = logits.softmax(dim=-1)
    return -(probs * (probs + 1e-8).log()).sum(dim=-1)


def tent_loss(predictions) -> torch.Tensor:
    """
    Compute the entropy loss from YOLOv8 raw predictions.

    YOLOv8 returns a tuple (boxes_tensor, ...) in training mode.
    The detection head output for a single scale is shaped:
      (batch, 4 + nc, H*W)
    We extract the class logits (last nc dimensions) across all scales.
    """
    # predictions can be a list of tensors from multi-scale heads
    if isinstance(predictions, (list, tuple)):
        entropy_terms = []
        for pred in predictions:
            if not isinstance(pred, torch.Tensor):
                continue
            if pred.ndim == 3:
                # shape: (batch, channels, anchors)
                # class scores are the last nc values in channel dim
                # split: first 4 = box regs, rest = class scores
                if pred.shape[1] > 4:
                    cls_logits = pred[:, 4:, :]          # (B, nc, anchors)
                    cls_logits = cls_logits.permute(0, 2, 1)  # (B, anchors, nc)
                    entropy_terms.append(softmax_entropy(cls_logits))
        if entropy_terms:
            return torch.cat([e.reshape(-1) for e in entropy_terms]).mean()
    # Fallback for single tensor
    return torch.tensor(0.0, requires_grad=True)


# ---------------------------------------------------------------------------
# TENT adapter class
# ---------------------------------------------------------------------------

class TENT:
    """
    Wraps a YOLO model for test-time adaptation via entropy minimisation.

    Usage:
        adapter = TENT(model, lr=0.001, steps=1)
        for batch in dataloader:
            adapted_preds = adapter.step(batch)   # adapts and predicts
    """

    def __init__(self, model: YOLO, lr: float = 0.001, steps: int = 1):
        self.model = model
        self.steps = steps
        self.lr = lr

        # Save original state so we can reset if needed
        self._original_state = copy.deepcopy(model.model.state_dict())

        # Configure the underlying nn.Module
        configure_model(model.model)
        params, param_names = collect_bn_params(model.model)
        print(f"[TENT] Adapting {len(params)} BN parameter tensors: {param_names[:6]}...")

        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    @torch.enable_grad()
    def step(self, batch_paths: list[str]) -> list:
        """
        Run `steps` gradient steps of entropy minimization then return predictions.
        batch_paths: list of image file paths (passed to YOLO inference).
        """
        for _ in range(self.steps):
            # Run forward pass in training mode to get raw head outputs for entropy
            # We use the underlying nn.Module directly
            results_raw = self.model.model(
                self._preprocess(batch_paths)
            )
            loss = tent_loss(results_raw)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Final inference pass (standard ultralytics API)
        return self.model.predict(batch_paths, verbose=False)

    def _preprocess(self, paths: list[str]) -> torch.Tensor:
        """Minimal preprocessing — delegate to ultralytics Dataset utilities."""
        from ultralytics.data.augment import LetterBox
        import numpy as np
        import cv2

        tensors = []
        lb = LetterBox(new_shape=(640, 640))
        for p in paths:
            img = cv2.imread(p)
            img = lb(image=img)
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()
            tensors.append(torch.from_numpy(img).float() / 255.0)
        batch = torch.stack(tensors)
        device = next(self.model.model.parameters()).device
        return batch.to(device)

    def reset(self):
        """Restore original BN statistics (for ablation / fresh adaptation)."""
        self.model.model.load_state_dict(self._original_state)
        configure_model(self.model.model)
