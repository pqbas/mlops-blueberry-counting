from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

N_ITERS = 10000
CI = 0.95
SEED = 0


def _ci(values: list[float], n_iters: int = N_ITERS, ci: float = CI, seed: int = SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = [mean(rng.choices(values, k=n)) for _ in range(n_iters)]
    samples.sort()
    lo = samples[int((1 - ci) / 2 * n_iters)]
    hi = samples[int((1 + ci) / 2 * n_iters) - 1]
    return lo, hi


def bootstrap_metrics(metrics_path: Path, exclude_prefixes: tuple[str, ...] = ()) -> dict:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    per = [v for v in data["per_video"] if not any(v["video"].startswith(p) for p in exclude_prefixes)]
    errs = [v["error_pct"] for v in per]
    abs_errs = [abs(e) for e in errs]
    mae_lo, mae_hi = _ci(abs_errs)
    bias_lo, bias_hi = _ci(errs)
    data["bootstrap"] = {
        "n_iters": N_ITERS,
        "ci": CI,
        "seed": SEED,
        "excluded_video_prefixes": list(exclude_prefixes),
        "n_videos_used": len(per),
        "mae": mean(abs_errs),
        "mae_ci": [round(mae_lo, 2), round(mae_hi, 2)],
        "bias": mean(errs),
        "bias_ci": [round(bias_lo, 2), round(bias_hi, 2)],
    }
    metrics_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data["bootstrap"]


def bootstrap_run(run_dir: Path, exclude_prefixes: tuple[str, ...] = ()) -> list[tuple[str, dict]]:
    metrics_paths = sorted(run_dir.rglob("metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"No hay metrics.json bajo {run_dir}")
    results = []
    for p in metrics_paths:
        label = p.parent.name if p.parent != run_dir else "(root)"
        b = bootstrap_metrics(p, exclude_prefixes=exclude_prefixes)
        results.append((label, b))
    return results
