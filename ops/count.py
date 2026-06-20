from __future__ import annotations

import json
from pathlib import Path

from ops.config import Config
from ops.dataset import load_manifest, video_path
from ops.strategies import count_video


def run(config: Config, output_dir: Path) -> None:
    """Operación count: corre la estrategia sobre cada video del dataset y
    escribe counts.json con el conteo por video."""
    video_ids = load_manifest(config.dataset)
    print(f"[count] estrategia={config.strategy} videos={len(video_ids)}")

    counts: dict[str, int] = {}
    for video_id in video_ids:
        path = video_path(config.dataset, video_id)
        render_path = output_dir / "renders" / f"{video_id}.mp4" if config.render else None
        n = count_video(config, path, render_path)
        counts[video_id] = n
        print(f"[count] {video_id}: {n}")

    payload = {
        "strategy": config.strategy,
        "type": config.type,
        "params": config.params,
        "videos": counts,
    }
    (output_dir / "counts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[count] counts.json escrito en {output_dir}")
