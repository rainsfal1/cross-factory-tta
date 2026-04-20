import copy

import torch
import torch.nn as nn
from ultralytics import YOLO


def configure_model_for_tent(model: nn.Module) -> nn.Module:
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            # Do NOT null running_mean/running_var — ultralytics fuses Conv+BN
            # on every val() call and needs these tensors to exist.
            # train() mode + track_running_stats=False is sufficient: BN uses
            # batch statistics and does not update the stored running stats.
    return model


def collect_bn_params(model: nn.Module):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for pnm, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pnm}")
    return params, names


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * (probs + 1e-8).log()).sum(dim=-1)


def tent_loss(predictions) -> torch.Tensor:
    # YOLOv8 train-mode returns a dict with 'scores': [B, num_classes, num_anchors]
    if isinstance(predictions, dict):
        scores = predictions.get("scores")
        if scores is not None and scores.requires_grad:
            cls_logits = scores.permute(0, 2, 1)  # [B, anchors, classes]
            return softmax_entropy(cls_logits).mean()
    # Fallback: list/tuple of raw head tensors
    if isinstance(predictions, (list, tuple)):
        entropy_terms = []
        for pred in predictions:
            if not isinstance(pred, torch.Tensor):
                continue
            if pred.ndim == 3 and pred.shape[1] > 4:
                cls_logits = pred[:, 4:, :].permute(0, 2, 1)
                entropy_terms.append(softmax_entropy(cls_logits))
        if entropy_terms:
            return torch.cat([e.reshape(-1) for e in entropy_terms]).mean()
    import warnings
    warnings.warn(
        f"tent_loss: unrecognized prediction format ({type(predictions)}), loss=0. "
        "No gradient will flow — check model forward output format.",
        RuntimeWarning, stacklevel=2,
    )
    return torch.tensor(0.0, requires_grad=True)


class TENT:
    def __init__(self, model: YOLO, lr: float = 0.001, steps: int = 1):
        self.model = model
        self.steps = steps
        self.lr    = lr
        self._original_state = copy.deepcopy(model.model.state_dict())
        configure_model_for_tent(model.model)
        params, _ = collect_bn_params(model.model)
        print(f"[TENT] Adapting {len(params)} BN parameters")
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        # Store the original forward before any wrapping
        self._orig_forward = model.model.forward

    def step_tensor(self, x: torch.Tensor) -> None:
        """Adapt BN params on a preprocessed image tensor (already on device)."""
        for _ in range(self.steps):
            preds = self._orig_forward(x, augment=False)
            loss  = tent_loss(preds)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
