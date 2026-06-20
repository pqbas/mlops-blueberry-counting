from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ops.config import Config


def count_video(
    config: "Config", video_path: Path, render_path: Path | None = None
) -> int:
    """Despacha el conteo de un video a la estrategia configurada.

    render_path, si se da, escribe un .mp4 anotado; lo soportan las tres
    estrategias.
    """
    if config.strategy == "line_crossing":
        from ops.strategies import line_crossing

        return line_crossing.count_video(
            video_path, config.detector_weights, config.params, config.device, render_path
        )

    if config.strategy == "tiled_crossing":
        from ops.strategies import tiled_crossing

        return tiled_crossing.count_video(
            video_path, config.detector_weights, config.params, config.device, render_path
        )

    from ops.strategies import embeddings

    return embeddings.count_video(
        video_path,
        config.detector_weights,
        config.encoder_weights,
        config.params,
        config.device,
        render_path,
    )
