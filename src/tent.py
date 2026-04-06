import copy

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2


def configure_model_for_tent(model: nn.Module) -> nn.Module:
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
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

    @torch.enable_grad()
    def step(self, batch_paths: list) -> list:
        for _ in range(self.steps):
            results_raw = self.model.model(self._preprocess(batch_paths))
            loss = tent_loss(results_raw)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.model.predict(batch_paths, verbose=False)

    def _preprocess(self, paths: list) -> torch.Tensor:
        tensors = []
        lb = LetterBox(new_shape=(640, 640))
        for p in paths:
            img = cv2.imread(str(p))
            img = lb(image=img)
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()
            tensors.append(torch.from_numpy(img).float() / 255.0)
        batch = torch.stack(tensors)
        return batch.to(next(self.model.model.parameters()).device)
