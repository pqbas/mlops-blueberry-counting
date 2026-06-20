from __future__ import annotations

import json
from pathlib import Path

EVALUATE_OUTPUT = "error_by_trial.png"
COUNT_OUTPUT = "count_by_trial.png"


def plot_experiment(run_dir: Path) -> Path:
    """Genera una imagen con una serie por trial del run.

    Para experimentos evaluate grafica el error % por video; para count
    grafica el conteo por video. Una línea por trial.
    """
    trials = _find_trials(run_dir)
    if not trials:
        raise FileNotFoundError(
            f"No se encontró ningún counts.json ni metrics.json en {run_dir}"
        )

    kind = trials[0][2]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, path, _kind in trials:
        videos, values = _read_series(path, kind)
        ax.plot(videos, values, marker="o", label=label)

    ax.set_xlabel("video")
    ax.set_ylabel("error %" if kind == "evaluate" else "conteo")
    title = "Error por video" if kind == "evaluate" else "Conteo por video"
    ax.set_title(f"{title} - {run_dir.name}")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    if kind == "evaluate":
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)

    out_path = run_dir / (EVALUATE_OUTPUT if kind == "evaluate" else COUNT_OUTPUT)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _find_trials(run_dir: Path) -> list[tuple[str, Path, str]]:
    """Localiza los archivos de resultados. Retorna [(label, path, kind), ...]
    donde kind es 'evaluate' o 'count'."""
    own = _trial_file(run_dir, run_dir.name)
    if own:
        return [own]
    trials: list[tuple[str, Path, str]] = []
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir():
            found = _trial_file(sub, sub.name)
            if found:
                trials.append(found)
    return trials


def _trial_file(directory: Path, label: str) -> tuple[str, Path, str] | None:
    metrics = directory / "metrics.json"
    if metrics.exists():
        return (label, metrics, "evaluate")
    counts = directory / "counts.json"
    if counts.exists():
        return (label, counts, "count")
    return None


def _read_series(path: Path, kind: str) -> tuple[list[str], list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if kind == "evaluate":
        rows = data["per_video"]
        videos = [r["video"] for r in rows]
        values = [r["error_pct"] if r["error_pct"] is not None else 0.0 for r in rows]
        return videos, values
    items = sorted(data["videos"].items())
    return [video for video, _ in items], [count for _, count in items]
