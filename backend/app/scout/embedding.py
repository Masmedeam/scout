from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import settings
from app.scout.observability import scout_op

MODEL_NAME = "scout-pil-equalized-grayscale-32x32-v1"


@scout_op("embed_image")
def embed_image(path: str | Path) -> np.ndarray:
    size = int(settings.SCOUT_EMBEDDING_DIM**0.5)
    if size * size != settings.SCOUT_EMBEDDING_DIM:
        raise ValueError("SCOUT_EMBEDDING_DIM must be a perfect square")

    with Image.open(path) as image:
        grayscale = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    values = np.asarray(grayscale, dtype=np.float32)
    lower, upper = np.percentile(values, [2, 98])
    if upper > lower:
        values = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    else:
        values = values / 255.0
    vector = values.reshape(-1)
    vector = vector - float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)
