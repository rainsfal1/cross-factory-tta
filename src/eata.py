"""
EATA: Efficient Anti-Forgetting Test-Time Adaptation (Niu et al., ICML 2022).
https://arxiv.org/abs/2204.02610

Two improvements over TENT:

  1. Dual sample filtering:
     - Reliability filter: exclude high-entropy samples (entropy > e_margin)
     - Redundancy filter: exclude samples whose softmax prediction is too
       similar to the running model probability vector (cosine sim < d_margin)
     This ensures only informative, non-redundant samples drive updates.

  2. Fisher regularization (EWC-style): penalizes deviation from
     source-domain BN parameters, weighted by pre-computed Fisher diagonal.
     Prevents catastrophic forgetting of source-domain class knowledge.

Unlike SAR, EATA uses plain SGD (no SAM). The anti-forgetting term is what
makes it more robust under severe shift.

e_margin default: log(1000)/2 - 1 = 2.454. Higher than SAR's paper formula
(0.4*log(17)=1.145) and above observed target-domain entropy (1.75-2.04),
so samples actually pass the filter.
"""
import copy
import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from src.tent import collect_bn_params, softmax_entropy


def _per_sample_entropy(predictions) -> torch.Tensor:
    """Mean entropy per image in batch [B], dict-aware for YOLOv8."""
    if isinstance(predictions, dict):
        scores = predictions.get("scores")
        if scores is not None:
            cls_logits = scores.permute(0, 2, 1)       # [B, anchors, C]
            return softmax_entropy(cls_logits).mean(dim=1)   # [B]
    if isinstance(predictions, (list, tuple)):
        terms = []
        for pred in predictions:
            if not isinstance(pred, torch.Tensor):
                continue
            if pred.ndim == 3 and pred.shape[1] > 4:
                cls_logits = pred[:, 4:, :].permute(0, 2, 1)  # [B, anchors, C]
                terms.append(softmax_entropy(cls_logits).mean(dim=1))  # [B]
        if terms:
            return torch.stack(terms).mean(dim=0)
    return torch.full((1,), float("inf"))


def _per_sample_softmax(predictions) -> torch.Tensor | None:
    """Mean softmax probability vector per image [B, C], for redundancy filter."""
    if isinstance(predictions, dict):
        scores = predictions.get("scores")
        if scores is not None:
            cls_logits = scores.permute(0, 2, 1)   # [B, anchors, C]
            return cls_logits.softmax(dim=-1).mean(dim=1)   # [B, C]
    if isinstance(predictions, (list, tuple)):
        terms = []
        for pred in predictions:
            if not isinstance(pred, torch.Tensor):
                continue
            if pred.ndim == 3 and pred.shape[1] > 4:
                cls_logits = pred[:, 4:, :].permute(0, 2, 1)  # [B, anchors, C]
                terms.append(cls_logits.softmax(dim=-1).mean(dim=1))  # [B, C]
        if terms:
            return torch.stack(terms).mean(dim=0)
    return None


def compute_fishers(
    model: YOLO,
    source_img_dir,
    device: torch.device,
    imgsz: int = 640,
    n_images: int = 200,
) -> dict:
    """Compute Fisher information diagonal on source-domain images.

    Fisher_i = E[(d log p / d theta_i)^2] ≈ mean squared gradient over
    source images. Used as EWC weights to penalize forgetting.

    Args:
        model: YOLO wrapper (model.model is the nn.Module)
        source_img_dir: path to source domain image directory
        device: torch device
        imgsz: image size (must match training)
        n_images: number of source images to use (more = better estimate)
    """
    from pathlib import Path
    lb = LetterBox(new_shape=(imgsz, imgsz))
    img_paths = sorted(Path(source_img_dir).glob("*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    img_paths = img_paths[:n_images]

    # Configure for adaptation — BN grad enabled, track_running_stats off
    model.model.train()
    for p in model.model.parameters():
        p.requires_grad_(False)
    for m in model.model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            m.track_running_stats = False

    params, names = collect_bn_params(model.model)
    fishers = {name: [torch.zeros_like(p.data), p.data.clone()] for name, p in zip(names, params)}

    orig_forward = model.model.forward

    print(f"  [EATA] computing Fisher on {len(img_paths)} source images...", end=" ", flush=True)
    for p_path in img_paths:
        img = cv2.imread(str(p_path))
        if img is None:
            continue
        img = lb(image=img)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        x = torch.from_numpy(img).float() / 255.0
        x = x.unsqueeze(0).to(device)

        preds = orig_forward(x, augment=False)
        ent = _per_sample_entropy(preds)
        loss = ent.mean()
        if loss.requires_grad:
            loss.backward()
            for name, param in zip(names, params):
                if param.grad is not None:
                    fishers[name][0] += param.grad.data ** 2 / len(img_paths)
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

    print(f"done")
    model.model.eval()
    return fishers


class EATA:
    def __init__(
        self,
        model: YOLO,
        lr: float = 0.001,
        steps: int = 1,
        e_margin: float = math.log(1000) / 2 - 1,  # 2.454
        d_margin: float = 0.05,
        fishers: dict | None = None,
        fisher_alpha: float = 2000.0,
    ):
        """
        Args:
            model: ultralytics YOLO wrapper
            lr: learning rate for SGD optimizer
            steps: gradient steps per batch
            e_margin: entropy threshold — samples above are excluded (reliability filter)
            d_margin: cosine similarity threshold — samples below are excluded (redundancy filter)
            fishers: pre-computed Fisher diagonal dict from compute_fishers()
            fisher_alpha: EWC regularization strength (0 = no Fisher penalty)
        """
        self.model = model
        self.steps = steps
        self.e_margin = e_margin
        self.d_margin = d_margin
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.current_model_probs: torch.Tensor | None = None

        self._original_state = copy.deepcopy(model.model.state_dict())
        self._configure()
        params, self._param_names = collect_bn_params(model.model)
        print(f"[EATA] Adapting {len(params)} BN parameters  "
              f"e_margin={e_margin:.3f}  d_margin={d_margin}  "
              f"fisher={'yes' if fishers else 'no'}  lr={lr}")
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        self._orig_forward = model.model.forward

    def _configure(self) -> None:
        self.model.model.train()
        for p in self.model.model.parameters():
            p.requires_grad_(False)
        for m in self.model.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.requires_grad_(True)
                m.track_running_stats = False

    def adapt(self, x: torch.Tensor, _debug: bool = False) -> None:
        """Adapt BN affine params using dual filtering + Fisher regularization.

        Args:
            x: [B, 3, H, W] preprocessed image tensor on device
        """
        for _ in range(self.steps):
            preds = self._orig_forward(x, augment=False)
            entropy = _per_sample_entropy(preds)        # [B]
            softmax_probs = _per_sample_softmax(preds)  # [B, C] or None

            # ── Filter 1: reliability — exclude high-entropy samples ─────────
            reliable_mask = entropy < self.e_margin
            n_reliable = int(reliable_mask.sum())
            if n_reliable == 0:
                return

            reliable_entropy = entropy[reliable_mask]   # [M]

            # ── Filter 2: redundancy — exclude near-duplicate predictions ────
            n_diverse = n_reliable
            if self.current_model_probs is not None and softmax_probs is not None:
                reliable_probs = softmax_probs[reliable_mask]   # [M, C]
                cos_sim = F.cosine_similarity(
                    self.current_model_probs.unsqueeze(0), reliable_probs, dim=1
                )  # [M]
                diverse_mask = torch.abs(cos_sim) < self.d_margin
                n_diverse = int(diverse_mask.sum())
                if _debug:
                    print(f"  [EATA debug] B={x.shape[0]}  entropy={entropy.tolist()}  "
                          f"e_margin={self.e_margin:.3f}  reliable={n_reliable}/{x.shape[0]}  "
                          f"cos_sim={cos_sim.tolist()}  d_margin={self.d_margin}  diverse={n_diverse}/{n_reliable}")
                reliable_entropy = reliable_entropy[diverse_mask]
                if reliable_entropy.size(0) == 0:
                    self._update_model_probs(softmax_probs[reliable_mask])
                    return
                self._update_model_probs(reliable_probs[diverse_mask])
            else:
                if _debug:
                    print(f"  [EATA debug] B={x.shape[0]}  entropy={entropy.tolist()}  "
                          f"e_margin={self.e_margin:.3f}  reliable={n_reliable}/{x.shape[0]}  "
                          f"no redundancy filter (first batch)")
                if softmax_probs is not None:
                    self._update_model_probs(softmax_probs[reliable_mask])

            # ── Entropy reweighting: lower entropy → higher weight ────────────
            coeff = 1.0 / torch.exp(reliable_entropy.clone().detach() - self.e_margin)
            loss = (reliable_entropy * coeff).mean()

            # ── Fisher regularization (EWC anti-forgetting term) ─────────────
            if self.fishers is not None and self.fisher_alpha > 0:
                ewc_loss = torch.tensor(0.0, device=x.device)
                for (name, param) in zip(self._param_names, [
                    p for p in self.model.model.parameters() if p.requires_grad
                ]):
                    if name in self.fishers:
                        fisher_diag, source_param = self.fishers[name]
                        ewc_loss = ewc_loss + self.fisher_alpha * (
                            fisher_diag * (param - source_param) ** 2
                        ).sum()
                loss = loss + ewc_loss

            self.optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()

    def _update_model_probs(self, new_probs: torch.Tensor) -> None:
        if new_probs.size(0) == 0:
            return
        mean_probs = new_probs.mean(dim=0).detach()
        if self.current_model_probs is None:
            self.current_model_probs = mean_probs
        else:
            self.current_model_probs = 0.9 * self.current_model_probs + 0.1 * mean_probs

    def reset(self) -> None:
        self.model.model.load_state_dict(self._original_state)
        self._configure()
        self.current_model_probs = None
