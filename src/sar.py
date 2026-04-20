"""
SAR: Sharpness-Aware and Reliable entropy minimization (Niu et al., 2023).
https://arxiv.org/abs/2302.12400

Extension of TENT with two improvements that address instability on small
or noisy test batches:

  1. Sharpness-aware minimization (SAM): instead of descending the entropy
     gradient directly, SAM finds a perturbation direction that maximally
     increases entropy, then descends from that perturbed point. This biases
     adaptation toward flat minima in the entropy landscape, which are more
     robust to domain shift than sharp minima found by SGD.

  2. Reliable sample filtering: before computing the gradient, samples whose
     entropy exceeds e_margin are discarded. High-entropy samples are ambiguous
     — their gradient signal is noisy and tends to destabilize BN affine params.
     Only low-entropy (confident) predictions are trusted for adaptation.

  3. Model reset: if the adapted model's loss diverges too far from an EMA
     anchor, the model is partially reset to prevent catastrophic forgetting.

Adapts BN affine parameters (gamma/beta) only, same as TENT.
"""
import copy
import math

import torch
import torch.nn as nn
from ultralytics import YOLO

from src.tent import collect_bn_params, softmax_entropy, tent_loss


class _SAM:
    """Minimal Sharpness-Aware Minimization optimizer (two-step update)."""

    def __init__(self, params: list, lr: float, rho: float = 0.05, momentum: float = 0.9):
        self.params = [p for p in params if p.requires_grad]
        self.rho = rho
        self._base = torch.optim.SGD(self.params, lr=lr, momentum=momentum)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb weights in the gradient direction (max-entropy step)."""
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return
        grad_norm = torch.stack([g.norm(2) for g in grads]).norm(2) + 1e-12
        scale = self.rho / grad_norm
        for p in self.params:
            if p.grad is None:
                continue
            e_w = p.grad.detach() * scale
            p.add_(e_w)
            p._sar_e_w = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Remove perturbation and apply the base SGD step."""
        for p in self.params:
            if hasattr(p, "_sar_e_w"):
                p.sub_(p._sar_e_w)
                del p._sar_e_w
        self._base.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self._base.zero_grad()


def _per_sample_entropy(predictions) -> torch.Tensor:
    """Mean entropy per image in the batch, shape [B]."""
    # YOLOv8 train-mode can return a dict with 'scores': [B, num_classes, num_anchors]
    if isinstance(predictions, dict):
        scores = predictions.get("scores")
        if scores is not None:
            cls_logits = scores.permute(0, 2, 1)   # [B, anchors, C]
            return softmax_entropy(cls_logits).mean(dim=1)  # [B]
    if isinstance(predictions, (list, tuple)):
        terms = []
        for pred in predictions:
            if not isinstance(pred, torch.Tensor):
                continue
            if pred.ndim == 3 and pred.shape[1] > 4:
                cls_logits = pred[:, 4:, :].permute(0, 2, 1)  # [B, anchors, C]
                ent = softmax_entropy(cls_logits)              # [B, anchors]
                terms.append(ent.mean(dim=1))                  # [B]
        if terms:
            return torch.stack(terms).mean(dim=0)             # [B]
    return torch.full((1,), float("inf"))


class SAR:
    def __init__(
        self,
        model: YOLO,
        lr: float = 0.00025,
        steps: int = 1,
        e_margin: float = 0.4 * math.log(17),   # 40% of ln(num_source_classes)
        reset_constant: float = 0.2,
        rho: float = 0.05,
    ):
        """
        Args:
            model: ultralytics YOLO wrapper
            lr: learning rate for SAM optimizer
            steps: gradient steps per batch
            e_margin: entropy threshold for sample filtering — samples with
                entropy > e_margin are excluded from the gradient update
            reset_constant: if mean entropy deviates from EMA anchor by this
                fraction, reset adapted weights to the EMA anchor
            rho: SAM perturbation radius (neighbourhood size for sharpness)
        """
        self.model = model
        self.steps = steps
        self.e_margin = e_margin
        self.reset_constant = reset_constant

        self._original_state = copy.deepcopy(model.model.state_dict())
        self._anchor_state   = copy.deepcopy(model.model.state_dict())
        self._ema_loss: float | None = None

        self._configure()
        params, _ = collect_bn_params(model.model)
        print(f"[SAR] Adapting {len(params)} BN parameters  "
              f"e_margin={e_margin:.3f}  rho={rho}")
        self.optimizer = _SAM(params, lr=lr, rho=rho)
        self._orig_forward = model.model.forward

    def _configure(self) -> None:
        """Same as TENT: train mode, freeze all, unfreeze BN affine."""
        self.model.model.train()
        for p in self.model.model.parameters():
            p.requires_grad_(False)
        for m in self.model.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.requires_grad_(True)
                m.track_running_stats = False

    def adapt(self, x: torch.Tensor, _debug: bool = False) -> None:
        """Adapt BN affine params on batch x using SAM + entropy filtering.

        Args:
            x: preprocessed image tensor [B, 3, H, W], on device, float32 in [0, 1]
        """
        for _ in range(self.steps):
            # ── Step 1: compute per-sample entropy, filter reliable samples ──
            preds = self._orig_forward(x, augment=False)
            entropy = _per_sample_entropy(preds)        # [B]

            if _debug:
                print(f"  [SAR debug] pred type={type(preds).__name__}  "
                      f"entropy={entropy.tolist()}  e_margin={self.e_margin:.3f}  "
                      f"reliable={int((entropy < self.e_margin).sum())}/{len(entropy)}")

            reliable = entropy < self.e_margin

            if reliable.sum() == 0:
                return  # all samples above margin — skip this batch

            loss = entropy[reliable].mean()
            if not loss.requires_grad:
                return

            # ── Step 2: SAM first step (perturb toward max-entropy direction) ─
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # ── Step 3: SAM second step (gradient at perturbed weights) ────────
            preds2   = self._orig_forward(x, augment=False)
            entropy2 = _per_sample_entropy(preds2)
            mask2    = entropy2 < self.e_margin
            if mask2.sum() > 0:
                loss2 = entropy2[mask2].mean()
                if loss2.requires_grad:
                    loss2.backward()
            self.optimizer.second_step(zero_grad=True)

            # ── Step 4: model reset check ───────────────────────────────────
            current_loss = float(loss.detach())
            if self._ema_loss is None:
                self._ema_loss = current_loss
            else:
                self._ema_loss = 0.9 * self._ema_loss + 0.1 * current_loss
                if abs(current_loss - self._ema_loss) > self.reset_constant * self._ema_loss:
                    self.model.model.load_state_dict(self._anchor_state)
                    self._configure()
                    self._ema_loss = None
                else:
                    # Update anchor EMA
                    for k in self._anchor_state:
                        cur = self.model.model.state_dict()[k]
                        self._anchor_state[k] = 0.999 * self._anchor_state[k] + 0.001 * cur

    def reset(self) -> None:
        """Restore source-domain weights."""
        self.model.model.load_state_dict(self._original_state)
        self._configure()
        self._ema_loss = None
