from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.core.config import settings
from app.scout.observability import scout_op

MODEL_NAME = settings.SCOUT_EMBEDDING_MODEL

_MODEL: CLIPModel | None = None
_PROCESSOR: Any | None = None
_DEVICE: torch.device | None = None


@scout_op("embed_image")
def embed_image(path: str | Path) -> np.ndarray:
    model, processor, device = _load_model()
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")

    inputs = processor(images=rgb_image, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
    vector = features[0].detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    if vector.shape != (settings.SCOUT_EMBEDDING_DIM,):
        raise ValueError(
            f"{MODEL_NAME} produced {vector.shape[0]} dims, "
            f"expected {settings.SCOUT_EMBEDDING_DIM}"
        )
    return vector.astype(np.float32)


def _load_model() -> tuple[CLIPModel, Any, torch.device]:
    global _DEVICE, _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None and _DEVICE is not None:
        return _MODEL, _PROCESSOR, _DEVICE

    device = _best_device()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    _DEVICE = device
    _MODEL = model
    _PROCESSOR = processor
    return model, processor, device


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
