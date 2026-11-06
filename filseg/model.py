"""Segmentation network and tiled / test-time-augmented prediction."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from . import IMAGE_SIZE

MEAN, STD = 0.45, 0.225


def build_model(arch: str = "Unet", encoder: str = "resnet34", weights: str | None = "imagenet",
                classes: int = 1) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    cls = getattr(smp, arch)
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=classes)


def compact_state_dict(model: torch.nn.Module) -> dict:
    """State dict with conv / linear weights (>= 2-D floating tensors) in fp16.

    1-D tensors (biases, BatchNorm statistics) stay in fp32. This halves the file
    size (ResNet34 U-Net: 98 MB -> 49 MB) so weights fit in a plain Git repository.
    """
    out = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        out[k] = v.half() if v.is_floating_point() and v.dim() >= 2 else v.clone()
    return out


def save_checkpoint(model: torch.nn.Module, path, **meta) -> None:
    """Save inference weights plus metadata (``arch``, ``encoder``, ``res``, ...)."""
    torch.save({"model": compact_state_dict(model), **meta}, path)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Rebuild the network described by a checkpoint and load its weights (fp32, eval mode)."""
    st = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(st["arch"], st["encoder"], weights=None)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st["model"].items()})
    return model.to(device).eval(), {k: v for k, v in st.items() if k != "model"}


def disk_stats(img_u8: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of ``img / 255`` inside the solar disk (full-resolution image)."""
    from .classical import disk_mask

    v = img_u8[disk_mask(img_u8, erode=16)].astype(np.float32) / 255.0
    if v.size < 1000:
        v = img_u8.astype(np.float32).ravel() / 255.0
    return float(v.mean()), max(float(v.std()), 1e-3)


def image_stats(img_u8: np.ndarray, norm: str) -> tuple[float, float]:
    """Normalisation statistics: fixed (``global``) or per image inside the disk (``disk``)."""
    if norm == "disk":
        return disk_stats(img_u8)
    if norm == "global":
        return MEAN, STD
    raise ValueError(f"unknown normalisation {norm!r}")


def to_tensor(img01: np.ndarray, stats: tuple[float, float] = (MEAN, STD)) -> torch.Tensor:
    """(H, W) float image in [0, 1] -> normalised (1, H, W) tensor, ``(img - mean) / std``."""
    return torch.from_numpy(((img01 - stats[0]) / stats[1]).astype(np.float32))[None]


def resize_image(img: np.ndarray, res: int) -> np.ndarray:
    if img.shape[0] == res:
        return img
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def predict_batch(model: torch.nn.Module, x: torch.Tensor, tta: int = 4) -> torch.Tensor:
    """Sigmoid probabilities averaged over dihedral TTA (1, 2, 4 flips; 8 = flips x rot90). x: (B, 1, H, W)."""
    ops = [(k, f) for k in (0, 1) for f in [(), (3,), (2,), (2, 3)]][:max(1, tta)]
    acc = None
    for k, f in ops:
        xi = torch.rot90(x, k, (2, 3)) if k else x
        xi = torch.flip(xi, f) if f else xi
        p = torch.sigmoid(model(xi).float())
        p = torch.flip(p, f) if f else p
        p = torch.rot90(p, -k, (2, 3)) if k else p
        acc = p if acc is None else acc + p
    return acc / len(ops)


@torch.no_grad()
def predict_images(models: list[torch.nn.Module], images: list[np.ndarray], res: int,
                   device: torch.device, tta: int = 4, batch: int = 4,
                   amp: bool = True, norm: str = "global") -> np.ndarray:
    """uint8 (2048, 2048) images -> float16 probabilities at ``res`` (N, res, res).

    Probabilities of several models (same resolution and normalisation) are averaged.
    """
    out = np.zeros((len(images), res, res), np.float16)
    use_amp = amp and device.type == "cuda"
    for s in range(0, len(images), batch):
        xb = torch.stack([to_tensor(resize_image(im, res).astype(np.float32) / 255.0, image_stats(im, norm))
                          for im in images[s:s + batch]]).to(device)
        acc = None
        with torch.autocast(device_type=device.type, enabled=use_amp):
            for m in models:
                p = predict_batch(m, xb, tta)
                acc = p if acc is None else acc + p
        out[s:s + len(xb)] = (acc / len(models))[:, 0].cpu().numpy().astype(np.float16)
    return out


def upsample(prob: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Bilinear upsampling of a probability map to full resolution (see ``postprocess.upsample``)."""
    from .postprocess import upsample as _up
    return _up(prob, size)
