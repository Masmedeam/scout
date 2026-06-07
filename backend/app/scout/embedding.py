from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.core.config import settings
from app.scout.observability import scout_op

MODEL_NAME = settings.SCOUT_EMBEDDING_MODEL
DINOV2_HF_MODEL = "facebook/dinov2-small"

_CLIP_MODEL: CLIPModel | None = None
_CLIP_PROCESSOR: Any | None = None
_DEVICE: torch.device | None = None


def active_embedding_model() -> str:
    if settings.SCOUT_EMBEDDING_BACKEND == "dinov2":
        return DINOV2_HF_MODEL
    return MODEL_NAME


@scout_op("embed_image")
def embed_image(path: str | Path) -> np.ndarray:
    if settings.SCOUT_EMBEDDING_BACKEND == "dinov2":
        return _embed_dinov2(path)
    return _embed_clip(path)


@scout_op("embed_images")
def embed_images(paths: list[str | Path]) -> list[np.ndarray]:
    if not paths:
        return []
    if settings.SCOUT_EMBEDDING_BACKEND == "dinov2":
        return [_embed_dinov2(path) for path in paths]
    return _embed_clip_batch(paths)


def release_embedding_cache() -> None:
    global _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE
    clear_embedding_cache()
    _CLIP_MODEL = None
    _CLIP_PROCESSOR = None
    _DEVICE = None


def clear_embedding_cache() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _embed_clip(path: str | Path) -> np.ndarray:
    vectors = _embed_clip_batch([path])
    return vectors[0]


def _embed_clip_batch(paths: list[str | Path]) -> list[np.ndarray]:
    model, processor, device = _load_clip()
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
    vectors = features.detach().cpu().numpy().astype(np.float32)
    return [_normalize_to_dim(vector) for vector in vectors]


@lru_cache(maxsize=1)
def _load_dinov2():  # noqa: ANN202
    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained(DINOV2_HF_MODEL)
    model = Dinov2Model.from_pretrained(DINOV2_HF_MODEL)
    model.eval()
    device = _best_device()
    model.to(device)
    return processor, model, device


def _embed_dinov2(path: str | Path) -> np.ndarray:
    processor, model, device = _load_dinov2()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
    return _normalize_to_dim(vector)


def _load_clip() -> tuple[CLIPModel, Any, torch.device]:
    global _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE
    if _CLIP_MODEL is not None and _CLIP_PROCESSOR is not None and _DEVICE is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE

    device = _best_device()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    _DEVICE = device
    _CLIP_MODEL = model
    _CLIP_PROCESSOR = processor
    return model, processor, device


def _normalize_to_dim(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    if vector.shape != (settings.SCOUT_EMBEDDING_DIM,):
        raise ValueError(
            f"{MODEL_NAME} produced {vector.shape[0]} dims, "
            f"expected {settings.SCOUT_EMBEDDING_DIM}"
        )
    return vector.astype(np.float32)


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
