from __future__ import annotations

import json
from pathlib import Path

from ops import groundtruth
from ops.config import Config
from ops.dataset import labels_dir, load_manifest, video_path
from ops.strategies import count_video


def run(config: Config, output_dir: Path) -> None:
    """Operación evaluate: corre la estrategia sobre cada video, calcula el
    ground truth y escribe metrics.json con el error por video y agregado."""
    video_ids = load_manifest(config.dataset)
    print(f"[evaluate] estrategia={config.strategy} videos={len(video_ids)}")

    per_video: list[dict] = []
    for video_id in video_ids:
        render_path = output_dir / "renders" / f"{video_id}.mp4" if config.render else None
        model_count = count_video(config, video_path(config.dataset, video_id), render_path)
        gt_count = _ground_truth(config, video_id)
        error_pct = (
            round(100.0 * (model_count - gt_count) / gt_count, 1) if gt_count > 0 else None
        )
        per_video.append(
            {"video": video_id, "model": model_count, "gt": gt_count, "error_pct": error_pct}
        )
        print(f"[evaluate] {video_id}: model={model_count} gt={gt_count} error={error_pct}%")

    abs_errors = [abs(r["error_pct"]) for r in per_video if r["error_pct"] is not None]
    signed_errors = [r["error_pct"] for r in per_video if r["error_pct"] is not None]
    metrics = {
        "strategy": config.strategy,
        "type": config.type,
        "params": config.params,
        "per_video": per_video,
        "mean_abs_error_pct": round(sum(abs_errors) / len(abs_errors), 2) if abs_errors else None,
        "max_abs_error_pct": round(max(abs_errors), 2) if abs_errors else None,
        "bias": round(sum(signed_errors) / len(signed_errors), 2) if signed_errors else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[evaluate] metrics.json escrito en {output_dir} "
        f"(mean_abs_error={metrics['mean_abs_error_pct']}% bias={metrics['bias']}%)"
    )


def _ground_truth(config: Config, video_id: str) -> int:
    """GT por cruce estricto de línea para todas las estrategias: solo cuenta
    los objetos que pasan la línea de un lado al otro."""
    labels = groundtruth.load_track_labels(labels_dir(config.dataset, video_id))
    params = config.params
    if config.strategy == "line_crossing":
        return groundtruth.count_line_crossing(
            labels, params["count_mode"], params["threshold"], params["direction"]
        )
    if config.strategy == "tiled_crossing":
        return groundtruth.count_line_crossing(labels, "horizontal", 0.5, params["direction"])
    # embeddings: la línea es el centro x de la región; el método asume left2right.
    x1, _y1, x2, _y2 = params["region"]
    return groundtruth.count_line_crossing(labels, "horizontal", (x1 + x2) / 2, "left2right")
