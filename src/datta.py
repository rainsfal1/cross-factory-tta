"""
DATTA: Detection-Aware Test-Time Adaptation.

Custom method for extreme domain shift where standard TTA fails.
Phase 2 showed: TENT/SAR/EATA get no useful signal from near-random target
predictions; DUA overwrites BN stats indiscriminately and degrades all domains.

Two stages:

  Stage 1 — Soft BN Warm-Up (parameter-free):
    Aligns BN running statistics toward the target domain before any gradient
    update. Uses Welford online mean across all test batches, then blends with
    source stats: new = alpha * source + (1 - alpha) * target.
    alpha=1.0 → pure source (no warmup); alpha=0.0 → pure target (= DUA aggressive).
    Blended running stats are used at eval time; Stage 2 forward passes use
    batch stats (track_running_stats=False), so the blended values are preserved.

  Stage 2 — Detection-Aware Entropy Minimization (gradient-based):
    Only anchors whose max class probability exceeds conf_threshold contribute
    to the loss. Avoids gradient starvation from low-confidence background
    predictions that dominate vanilla TENT under extreme shift.
    Loss: mean(conf_i * entropy_i) for anchors where conf_i > conf_threshold.

Ablation flags use_stage1 / use_stage2 control which stages are active,
enabling the S1-only / S2-only / full ablation table.
"""
import copy

import torch
import torch.nn as nn
from ultralytics import YOLO

from src.tent import collect_bn_params, softmax_entropy


class DATTA:
    def __init__(
        self,
        model: YOLO,
        alpha: float = 0.5,
        conf_threshold: float = 0.25,
        lr: float = 0.005,
        steps: int = 1,
        use_stage1: bool = True,
        use_stage2: bool = True,
    ):
        """
        Args:
            model: ultralytics YOLO wrapper (weights loaded, eval mode)
            alpha: BN warmup mixing coefficient [0, 1].
                   1.0 = keep source stats (no-op); 0.0 = full target stats.
            conf_threshold: min max-class-prob for an anchor to enter the loss
            lr: SGD learning rate for Stage 2 BN affine params
            steps: gradient steps per adapt() call
            use_stage1: enable Soft BN Warm-Up
            use_stage2: enable Detection-Aware Entropy Minimization
        """
        self.model = model
        self.alpha = alpha
        self.conf_threshold = conf_threshold
        self.steps = steps
        self.use_stage1 = use_stage1
        self.use_stage2 = use_stage2

        self._original_state = copy.deepcopy(model.model.state_dict())
        self._orig_forward = model.model.forward

        # Save source BN running stats before any modification
        self._source_bn_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            name: (m.running_mean.clone(), m.running_var.clone())
            for name, m in model.model.named_modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
            and m.running_mean is not None
        }

        if use_stage2:
            self._configure_for_adaptation()
            params, self._param_names = collect_bn_params(model.model)
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
            print(f"[DATTA] Adapting {len(params)} BN params  "
                  f"alpha={alpha}  conf_thr={conf_threshold}  lr={lr}  steps={steps}  "
                  f"stage1={'on' if use_stage1 else 'off'}  stage2=on")
        else:
            model.model.eval()
            print(f"[DATTA] Stage 1 only  alpha={alpha}  stage2=off")

    def _configure_for_adaptation(self) -> None:
        """TENT-style setup: train mode, freeze all, unfreeze BN affine.
        track_running_stats=False keeps running stats frozen (blended values
        from Stage 1 are preserved for val() but not overwritten during adapt).
        """
        self.model.model.train()
        for p in self.model.model.parameters():
            p.requires_grad_(False)
        for m in self.model.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.requires_grad_(True)
                m.track_running_stats = False

    def warmup(self, batches: list[torch.Tensor]) -> None:
        """Stage 1: accumulate target BN stats via Welford averaging, blend with source.

        Call once with all test batches before the adapt() loop.
        Welford online mean: momentum = 1/(t+1) → equal-weighted average across batches.
        Note: batch variances are averaged (not pooled), which is an approximation —
        accurate enough for a warm-up initialization.

        Args:
            batches: list of [B, 3, H, W] tensors, all target-domain images
        """
        if not self.use_stage1 or not batches:
            return

        # Temporarily put all BN layers in train mode with running stat tracking
        for m in self.model.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.training = True
                m.track_running_stats = True

        # Forward all batches with Welford-style decaying momentum
        with torch.no_grad():
            for t, x in enumerate(batches):
                mom = 1.0 / (t + 1)
                for m in self.model.model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.momentum = mom
                self._orig_forward(x, augment=False)

        # Blend: new_stat = alpha * source + (1 - alpha) * accumulated_target
        for name, m in self.model.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and name in self._source_bn_stats:
                src_mean, src_var = self._source_bn_stats[name]
                m.running_mean.data = self.alpha * src_mean + (1.0 - self.alpha) * m.running_mean
                m.running_var.data = self.alpha * src_var + (1.0 - self.alpha) * m.running_var

        # Restore configuration — blended running stats persist for val()
        if self.use_stage2:
            self._configure_for_adaptation()
        else:
            self.model.model.eval()

    def _detection_aware_loss(self, predictions, _debug: bool = False) -> torch.Tensor | None:
        """Confidence-weighted entropy over anchors above conf_threshold.

        Returns None if no anchor clears the threshold (skip update).
        """
        if isinstance(predictions, dict):
            scores = predictions.get("scores")
            if scores is not None and scores.requires_grad:
                cls_logits = scores.permute(0, 2, 1)   # [B, anchors, C]
                probs = cls_logits.softmax(dim=-1)       # [B, anchors, C]
                conf = probs.max(dim=-1).values           # [B, anchors]
                entropy = softmax_entropy(cls_logits)     # [B, anchors]
                mask = conf > self.conf_threshold
                if _debug:
                    total = mask.numel()
                    passed = int(mask.sum())
                    pct = 100.0 * passed / max(total, 1)
                    print(f"  [DATTA debug] B={scores.shape[0]}  "
                          f"anchors_per_img={total // scores.shape[0]}  "
                          f"passed={passed}/{total} ({pct:.2f}%)  "
                          f"conf: max={float(conf.max()):.3f} "
                          f"mean={float(conf.mean()):.3f} "
                          f"p95={float(conf.flatten().topk(max(1, total//20)).values[-1]):.3f}  "
                          f"thr={self.conf_threshold}")
                if not mask.any():
                    return None
                return (conf[mask] * entropy[mask]).mean()

        if isinstance(predictions, (list, tuple)):
            terms = []
            for pred in predictions:
                if not isinstance(pred, torch.Tensor) or pred.ndim != 3 or pred.shape[1] <= 4:
                    continue
                cls_logits = pred[:, 4:, :].permute(0, 2, 1)
                probs = cls_logits.softmax(dim=-1)
                conf = probs.max(dim=-1).values
                entropy = softmax_entropy(cls_logits)
                mask = conf > self.conf_threshold
                if mask.any():
                    terms.append((conf[mask] * entropy[mask]).mean())
            if terms:
                return torch.stack(terms).mean()

        return None

    def adapt(self, x: torch.Tensor, _debug: bool = False) -> None:
        """Stage 2: one batch of confidence-weighted entropy minimization.

        Args:
            x: [B, 3, H, W] preprocessed image tensor on device
            _debug: if True, print mask stats (fraction of anchors above conf_threshold)
        """
        if not self.use_stage2:
            return
        for _ in range(self.steps):
            preds = self._orig_forward(x, augment=False)
            loss = self._detection_aware_loss(preds, _debug=_debug)
            if loss is None:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def reset(self) -> None:
        """Restore source-domain weights."""
        self.model.model.load_state_dict(self._original_state)
        if self.use_stage2:
            self._configure_for_adaptation()
        else:
            self.model.model.eval()
