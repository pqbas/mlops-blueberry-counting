from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from ops.config import Config, ConfigError, expand_configs, load_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ops", description="Counting MLOps runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Ejecuta un experimento desde un YAML")
    run_p.add_argument("yaml", type=Path, help="Ruta al YAML del experimento")
    plot_p = sub.add_parser("plot", help="Genera la imagen por trial de un run")
    plot_p.add_argument("run_dir", type=Path, help="Ruta runs/<name>/<timestamp>")
    boot_p = sub.add_parser("bootstrap", help="Calcula IC95% por bootstrap sobre per_video y lo agrega al metrics.json")
    boot_p.add_argument("run_dir", type=Path, help="Ruta runs/<name>/<timestamp>")
    boot_p.add_argument("--exclude", nargs="*", default=[], help="Prefijos de video_id a excluir del bootstrap")
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args.yaml)
    if args.command == "plot":
        return _cmd_plot(args.run_dir)
    if args.command == "bootstrap":
        return _cmd_bootstrap(args.run_dir, tuple(args.exclude))
    parser.error(f"Comando desconocido: {args.command}")
    return 2


def _cmd_run(yaml_path: Path) -> int:
    try:
        data = load_yaml(yaml_path)
        variants = expand_configs(data)
    except ConfigError as exc:
        print(f"[ops] Error de configuración: {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_cfg = variants[0][0]
    base_dir = Path("runs") / first_cfg.name / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(yaml_path, base_dir / "config.yaml")

    runs: list[tuple[Config, Path]] = []
    for cfg, suffix in variants:
        if suffix is None:
            out_dir = base_dir
        else:
            out_dir = base_dir / suffix
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False), encoding="utf-8")
        runs.append((cfg, out_dir))

    print(
        f"[ops] Experimento '{first_cfg.name}' tipo={first_cfg.type} "
        f"estrategia={first_cfg.strategy} runtime={first_cfg.runtime} trials={len(runs)}"
    )
    print(f"[ops] Outputs: {base_dir}")

    runtime_mod = importlib.import_module(f"ops.runtimes.{_runtime_module(first_cfg.runtime)}")
    runtime_mod.execute(runs)
    print(f"[ops] Listo. Resultados en {base_dir}")
    return 0


def _cmd_plot(run_dir: Path) -> int:
    from ops.plot import plot_experiment

    if not run_dir.is_dir():
        print(f"[ops] Directorio no encontrado: {run_dir}", file=sys.stderr)
        return 2
    try:
        out_path = plot_experiment(run_dir)
    except FileNotFoundError as exc:
        print(f"[ops] {exc}", file=sys.stderr)
        return 2
    print(f"[ops] Imagen generada: {out_path}")
    _open_image(out_path)
    return 0


def _cmd_bootstrap(run_dir: Path, exclude_prefixes: tuple[str, ...]) -> int:
    from ops.bootstrap import bootstrap_run

    if not run_dir.is_dir():
        print(f"[ops] Directorio no encontrado: {run_dir}", file=sys.stderr)
        return 2
    try:
        results = bootstrap_run(run_dir, exclude_prefixes=exclude_prefixes)
    except FileNotFoundError as exc:
        print(f"[ops] {exc}", file=sys.stderr)
        return 2
    print(f"[ops] Bootstrap n=10000 CI=95% excluyendo={list(exclude_prefixes)}")
    print(f"{'trial':<60} {'mae':>6}  {'mae IC95':>20}  {'bias':>7}  {'bias IC95':>20}")
    for label, b in results:
        mae_str = f"[{b['mae_ci'][0]:+.2f}, {b['mae_ci'][1]:+.2f}]"
        bias_str = f"[{b['bias_ci'][0]:+.2f}, {b['bias_ci'][1]:+.2f}]"
        print(f"{label:<60} {b['mae']:>5.2f}%  {mae_str:>20}  {b['bias']:>+6.2f}%  {bias_str:>20}")
    return 0


def _open_image(path: Path) -> None:
    opener = shutil.which("xdg-open")
    if opener is None:
        return
    subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _runtime_module(runtime: str) -> str:
    return "local" if runtime == "localhost" else runtime


if __name__ == "__main__":
    raise SystemExit(main())
