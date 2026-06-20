from __future__ import annotations

from pathlib import Path

import yaml

from ops.config import ConfigError


def load_manifest(manifest_path: Path) -> list[str]:
    """Lee manifest.yaml y devuelve la lista ordenada de video_ids del dataset.

    Formato esperado:

        videos:
          - vid_001
          - vid_002

    También se acepta la forma `- id: vid_001` por entrada.
    """
    if not manifest_path.exists():
        raise ConfigError(f"Manifest no encontrado: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Manifest inválido en {manifest_path}: se esperaba un mapping")
    videos = data.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ConfigError(f"Manifest {manifest_path}: 'videos' debe ser una lista no vacía")

    ids: list[str] = []
    for entry in videos:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict) and "id" in entry:
            ids.append(str(entry["id"]))
        else:
            raise ConfigError(f"Manifest {manifest_path}: entrada de video inválida: {entry!r}")
    return ids


def video_path(manifest_path: Path, video_id: str) -> Path:
    """Ruta al archivo .mp4 del video dentro del dataset."""
    return manifest_path.parent / "videos" / f"{video_id}.mp4"


def labels_dir(manifest_path: Path, video_id: str) -> Path:
    """Directorio con los labels por frame del video dentro del dataset."""
    return manifest_path.parent / "labels" / video_id
