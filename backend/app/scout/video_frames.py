"""Extract frames from drone video via ffmpeg."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedFrame:
    frame_id: str
    file_path: Path
    timestamp_s: float


def extract_frames(
    *,
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
) -> list[ExtractedFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%04d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(pattern),
    ]
    subprocess.run(command, check=True, capture_output=True)

    frames: list[ExtractedFrame] = []
    for index, frame_path in enumerate(sorted(output_dir.glob("frame_*.jpg")), start=1):
        timestamp_s = (index - 1) / fps
        frames.append(
            ExtractedFrame(
                frame_id=f"frame-{index:04d}",
                file_path=frame_path,
                timestamp_s=timestamp_s,
            )
        )
    return frames
