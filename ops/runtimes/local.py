from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from ops.config import Config


def execute(runs: list[tuple[Config, Path]]) -> None:
    """Ejecuta cada trial en un subproceso aislado (spawn), uno por uno.

    El aislamiento evita arrastrar estado de CUDA / ultralytics entre trials.
    """
    ctx = mp.get_context("spawn")
    for cfg, out_dir in runs:
        proc = ctx.Process(target=_run_one, args=(cfg, out_dir))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Trial falló (exitcode={proc.exitcode}): {out_dir}")


def _run_one(cfg: Config, out_dir: Path) -> None:
    from ops import count, evaluate

    if cfg.type == "count":
        count.run(cfg, out_dir)
    elif cfg.type == "evaluate":
        evaluate.run(cfg, out_dir)
    else:
        raise TypeError(f"Tipo de operación no soportado: {cfg.type}")
