from __future__ import annotations

from pathlib import Path
from typing import Tuple
import hashlib
import shutil
import urllib.request

import numpy as np


_BUILTIN_UNET_NAME = "affable_shark_nuclei_unet_torchscript.pt"
_BUILTIN_UNET_SHA256 = "8410950508655a300793b389c815dc30b1334062fc1dadb1e15e55a93cbb99a0"
_BUILTIN_UNET_URLS = (
    "https://hypha.aicell.io/bioimage-io/artifacts/affable-shark/files/weights-torchscript.pt",
    "https://zenodo.org/records/11085220/files/weights-torchscript.pt?download=1",
    "https://zenodo.org/record/6647674/files/weights-torchscript.pt?download=1",
)


def _read_section_image(section, channel: int = 0) -> np.ndarray:
    try:
        from PyReconstruct.modules.backend.view.channel_utils import read_section_channels
        channels = read_section_channels(section)
        if not channels:
            raise ValueError("No image channels found")
        idx = max(0, min(int(channel), len(channels) - 1))
        arr = np.asarray(channels[idx])
    except Exception:
        if section.series.src_dir.endswith("zarr"):
            import zarr
            arr = np.asarray(zarr.open(section.src_fp, mode="r")[:])
        else:
            import cv2
            arr = cv2.imread(str(section.src_fp), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(f"Could not read image: {section.src_fp}")
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        while arr.ndim > 2 and 1 in arr.shape:
            arr = np.squeeze(arr, axis=next(i for i, s in enumerate(arr.shape) if s == 1))
        if arr.ndim == 3:
            if arr.shape[-1] <= 8:
                idx = max(0, min(int(channel), arr.shape[-1] - 1))
                arr = arr[..., idx]
            elif arr.shape[0] <= 8:
                idx = max(0, min(int(channel), arr.shape[0] - 1))
                arr = arr[idx]
            else:
                arr = arr[..., 0]
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr, axis=next(i for i, s in enumerate(arr.shape) if s == 1))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image channel, got shape {arr.shape}")
    return arr


def _normalize_image(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.float32)
    vals = x[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    x = (x - lo) / (hi - lo + 1e-8)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _standardize_image(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.float32)
    vals = x[finite]
    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-6:
        std = 1.0
    x = (x - mean) / std
    x[~finite] = 0.0
    return x.astype(np.float32)


def _color_for_index(i: int) -> Tuple[int, int, int]:
    return int((37 * i + 80) % 256), int((91 * i + 120) % 256), int((53 * i + 180) % 256)


def _contour_to_points(contour: np.ndarray, img_height: int, mag: float):
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return []
    return [(float(x) * mag, float(img_height - y) * mag) for x, y in pts]


def _labels_to_traces(series, section_num: int, labels: np.ndarray, prefix: str, min_area: int, group_name: str) -> int:
    import cv2
    from PyReconstruct.modules.datatypes import Trace

    section = series.loadSection(section_num)
    img_height = int(labels.shape[0])
    mag = section.mag
    created = 0
    ids = [int(i) for i in np.unique(labels) if int(i) != 0]

    for label_id in ids:
        mask = (labels == label_id).astype(np.uint8)
        if int(mask.sum()) < int(min_area):
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < float(min_area):
                continue
            epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
            contour = cv2.approxPolyDP(contour, epsilon, True)
            points = _contour_to_points(contour, img_height, mag)
            if len(points) < 3:
                continue
            created += 1
            trace_name = f"{prefix}{created:05d}"
            trace = Trace(trace_name, _color_for_index(created), closed=True)
            trace.points = points
            trace.fill_mode = ("transparent", "unselected")
            trace.tags.add(group_name)
            section.addTrace(trace, log_event=True)
            series.object_groups.add(group_name, trace_name)

    section.save(update_series_data=True)
    series.save()
    return created


class _ConvBlock(__import__("torch").nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        import torch.nn as nn
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(__import__("torch").nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        import torch
        import torch.nn as nn
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)
        self._torch = torch

    def _crop_or_pad(self, x, ref):
        import torch.nn.functional as F
        dy = ref.shape[-2] - x.shape[-2]
        dx = ref.shape[-1] - x.shape[-1]
        if dy > 0 or dx > 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        if dy < 0 or dx < 0:
            y0 = (-dy) // 2
            x0 = (-dx) // 2
            x = x[..., y0:y0 + ref.shape[-2], x0:x0 + ref.shape[-1]]
        return x

    def forward(self, x):
        import torch
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self._crop_or_pad(self.up3(b), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._crop_or_pad(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._crop_or_pad(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def _resolve_device(device: str):
    import torch
    device = str(device or "cpu").lower()
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _builtin_model_dirs():
    here = Path(__file__).resolve().parent
    return (
        here / "models",
        Path.home() / ".pyreconstruct" / "plugins" / "models",
    )


def _builtin_model_path() -> Path:
    for folder in _builtin_model_dirs():
        path = folder / _BUILTIN_UNET_NAME
        if path.is_file() and _sha256(path) == _BUILTIN_UNET_SHA256:
            return path
    target_dir = Path.home() / ".pyreconstruct" / "plugins" / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _BUILTIN_UNET_NAME
    if target.is_file() and _sha256(target) == _BUILTIN_UNET_SHA256:
        return target
    errors = []
    for url in _BUILTIN_UNET_URLS:
        tmp = target.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url, timeout=90) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if _sha256(tmp) != _BUILTIN_UNET_SHA256:
                raise ValueError("downloaded model checksum did not match")
            tmp.replace(target)
            return target
        except Exception as e:
            errors.append(f"{url}: {e}")
            try:
                tmp.unlink()
            except Exception:
                pass
    raise RuntimeError("Could not download the built-in fluorescence nuclei U-Net. Connect to the internet and try again. Details: " + " | ".join(errors))


def _load_unet(checkpoint_path: str | None, device, model_source: str = "builtin"):
    import torch
    source = str(model_source or "builtin").lower()
    path = _builtin_model_path() if source == "builtin" else Path(str(checkpoint_path or ""))
    if source != "builtin" and not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        model = torch.jit.load(str(path), map_location=device)
        model.to(device)
        model.eval()
        return model, source
    except Exception:
        if source == "builtin":
            raise
    ckpt = torch.load(str(path), map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    out_channels = 1
    if "out.weight" in state:
        out_channels = int(state["out.weight"].shape[0])
    model = UNet2D(in_channels=1, out_channels=out_channels, base=32).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, source


def _model_probabilities(img: np.ndarray, checkpoint_path: str | None, device: str, model_source: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    dev = _resolve_device(device)
    model, source = _load_unet(checkpoint_path, dev, model_source)
    x = _standardize_image(img) if source == "builtin" else _normalize_image(img)
    h, w = x.shape
    target_h = max(64, ((h + 15) // 16) * 16)
    target_w = max(64, ((w + 15) // 16) * 16)
    pad_h = target_h - h
    pad_w = target_w - w
    tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(dev)
    if pad_h or pad_w:
        mode = "reflect" if h > 1 and w > 1 else "replicate"
        tensor = F.pad(tensor, [0, pad_w, 0, pad_h], mode=mode)
    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        while out.ndim > 4:
            out = out[0]
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 2:
            out = out.unsqueeze(0)
        out = out[..., :h, :w]
        if float(out.min()) < 0.0 or float(out.max()) > 1.0:
            out = torch.sigmoid(out)
        return out.detach().cpu().numpy().astype(np.float32)


def _remove_small_labels(labels: np.ndarray, min_area: int) -> np.ndarray:
    labels = labels.astype(np.int32, copy=True)
    for label_id in [int(i) for i in np.unique(labels) if int(i) != 0]:
        if int((labels == label_id).sum()) < int(min_area):
            labels[labels == label_id] = 0
    return labels


def _binary_to_labels(binary: np.ndarray) -> np.ndarray:
    import cv2
    _, labels = cv2.connectedComponents(binary.astype(np.uint8))
    return labels.astype(np.int32)


def _probabilities_to_labels(prob: np.ndarray, threshold: float, boundary_threshold: float, min_area: int) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim == 2:
        return _remove_small_labels(_binary_to_labels(prob >= float(threshold)), min_area)
    if prob.shape[0] == 1:
        return _remove_small_labels(_binary_to_labels(prob[0] >= float(threshold)), min_area)
    boundary = prob[0]
    foreground = prob[-1]
    mask = foreground >= float(threshold)
    seeds = mask & (boundary < float(boundary_threshold))
    seeds = ndi.binary_opening(seeds, iterations=1)
    markers, _ = ndi.label(seeds)
    if int(markers.max()) == 0:
        return _remove_small_labels(_binary_to_labels(mask), min_area)
    labels = watershed(boundary, markers.astype(np.int32), mask=mask)
    return _remove_small_labels(labels.astype(np.int32), min_area)


def run_unet_segmentation_on_section(series, section_num: int, checkpoint_path: str | None = None, model_source: str = "builtin", channel: int = 0, prefix: str = "unet_roi_", threshold: float = 0.5, boundary_threshold: float = 0.5, min_area: int = 25, device: str = "cpu") -> int:
    section = series.loadSection(section_num)
    img = _read_section_image(section, channel=channel)
    prob = _model_probabilities(img, checkpoint_path, device, model_source)
    labels = _probabilities_to_labels(prob, threshold, boundary_threshold, min_area)
    return _labels_to_traces(series, section_num, labels, prefix, min_area, "seg_unet")


def _load_cellpose_model(gpu: bool = False, model_source: str = "builtin", model_path: str | None = None):
    from pathlib import Path
    from cellpose import models

    source = str(model_source or "builtin").lower()
    if source == "custom":
        path = Path(str(model_path or "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Custom Cellpose/SAM model not found: {path}")
        return models.CellposeModel(gpu=bool(gpu), pretrained_model=str(path))

    try:
        return models.CellposeModel(gpu=bool(gpu), pretrained_model="cpsam")
    except Exception:
        return models.CellposeModel(gpu=bool(gpu), pretrained_model="cpsam_v2")


def run_cellpose_sam_segmentation_on_section(series, section_num: int, prefix: str = "cpsam_roi_", diameter=None, min_area: int = 25, gpu: bool = False, channel: int = 0, model_source: str = "builtin", model_path: str | None = None) -> int:
    section = series.loadSection(section_num)
    img = _read_section_image(section, channel=channel)
    img = _normalize_image(img)
    model = _load_cellpose_model(gpu=gpu, model_source=model_source, model_path=model_path)
    try:
        result = model.eval(img, channels=[0, 0], diameter=diameter, min_size=int(min_area))
    except TypeError:
        result = model.eval(img, channels=[0, 0], diameter=diameter)
    masks = result[0] if isinstance(result, (tuple, list)) else result
    labels = np.asarray(masks).astype(np.int32)
    return _labels_to_traces(series, section_num, labels, prefix, min_area, "seg_cellpose_sam")
