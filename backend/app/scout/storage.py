import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings


def storage_root() -> Path:
    root = Path(settings.SCOUT_STORAGE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_storage_dirs() -> dict[str, Path]:
    root = storage_root()
    dirs = {
        "rasters": root / "rasters",
        "patches": root / "patches",
        "queries": root / "queries",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_suffix(filename: str | None) -> str:
    if not filename:
        return ".jpg"
    suffix = Path(filename).suffix.lower()
    return suffix if suffix else ".jpg"


def save_upload(upload: UploadFile, folder: str) -> Path:
    dirs = ensure_storage_dirs()
    path = dirs[folder] / f"{uuid.uuid4()}{safe_suffix(upload.filename)}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path
