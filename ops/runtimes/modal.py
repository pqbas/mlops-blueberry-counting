from __future__ import annotations

import copy
import os
import time
from pathlib import Path

os.environ.setdefault("MODAL_PROFILE", "pcubasm1")

import modal

from ops.config import Config

APP_NAME = "counting-mlops"
DATASETS_VOLUME = "counting-datasets"
MODELS_VOLUME = "counting-models"
OUTPUTS_VOLUME = "counting-runs"
DATASETS_MOUNT = "/datasets"
MODELS_MOUNT = "/models"
OUTPUTS_MOUNT = "/outputs"
GPU = "T4"
TIMEOUT_SECONDS = 7200
POLL_INTERVAL_SECONDS = 5

_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("ultralytics==8.4.51", "pyyaml==6.0.3", "lap>=0.5.12")
    .add_local_python_source("ops")
)

_dataset_vol = modal.Volume.from_name(DATASETS_VOLUME, create_if_missing=True)
_models_vol = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)
_outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=_image,
    gpu=GPU,
    volumes={
        DATASETS_MOUNT: _dataset_vol,
        MODELS_MOUNT: _models_vol,
        OUTPUTS_MOUNT: _outputs_vol,
    },
    timeout=TIMEOUT_SECONDS,
)
def _run_experiment(config_raw: dict, output_subdir: str) -> None:
    from ops import count as count_mod
    from ops import evaluate as evaluate_mod
    from ops.config import parse_dict

    cfg = parse_dict(config_raw)
    out_root = Path(OUTPUTS_MOUNT) / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    if cfg.type == "count":
        count_mod.run(cfg, out_root)
    elif cfg.type == "evaluate":
        evaluate_mod.run(cfg, out_root)
    else:
        raise TypeError(f"Tipo de operación no soportado: {cfg.type}")

    _outputs_vol.commit()


def execute(runs: list[tuple[Config, Path]]) -> None:
    if not runs:
        return

    first_cfg = runs[0][0]
    dataset_dir = first_cfg.dataset.parent
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_dir}")
    dataset_name = dataset_dir.name
    _ensure_dataset_volume(dataset_dir, dataset_name)

    weight_paths = {cfg.detector_weights for cfg, _ in runs}
    weight_paths |= {cfg.encoder_weights for cfg, _ in runs if cfg.encoder_weights is not None}
    _ensure_models_volume(weight_paths)

    print(f"[modal] Job remoto GPU={GPU} trials={len(runs)}")
    with modal.enable_output():
        with app.run():
            pending: dict = {}
            for cfg, out_dir in runs:
                raw_remote = _rewrite_for_remote(cfg, dataset_name)
                subdir = str(out_dir.relative_to(Path("runs")))
                call = _run_experiment.spawn(raw_remote, subdir)
                pending[call] = (out_dir, subdir)

            while pending:
                finished = []
                for call, (out_dir, subdir) in pending.items():
                    try:
                        call.get(timeout=0)
                    except TimeoutError:
                        continue
                    finished.append(call)
                    print(f"[modal] Trial terminado, descargando {out_dir}")
                    _download_dir(subdir, out_dir)
                for call in finished:
                    del pending[call]
                if pending:
                    time.sleep(POLL_INTERVAL_SECONDS)


def _ensure_dataset_volume(local_dir: Path, name: str) -> None:
    if _volume_has_entries(_dataset_vol, name):
        print(f"[modal] Dataset '{name}' ya está en el volumen, omito upload")
        return
    print(f"[modal] Subiendo dataset '{name}' al volumen (puede tardar)")
    with _dataset_vol.batch_upload(force=False) as batch:
        batch.put_directory(str(local_dir), name)


def _ensure_models_volume(weight_paths: set[Path]) -> None:
    print(f"[modal] Subiendo {len(weight_paths)} archivo(s) de pesos al volumen")
    with _models_vol.batch_upload(force=True) as batch:
        for path in weight_paths:
            if not path.exists():
                raise FileNotFoundError(f"Pesos no encontrados: {path}")
            batch.put_file(str(path), path.name)


def _volume_has_entries(volume, name: str) -> bool:
    try:
        return any(True for _ in volume.iterdir(name))
    except Exception:
        return False


def _rewrite_for_remote(config: Config, dataset_name: str) -> dict:
    """Reescribe las rutas locales del config a las rutas montadas en Modal."""
    raw = copy.deepcopy(config.raw)
    original_dataset = Path(raw["dataset"])
    raw["dataset"] = f"{DATASETS_MOUNT}/{dataset_name}/{original_dataset.name}"

    model = raw["model"]
    model["detector_weights"] = f"{MODELS_MOUNT}/{Path(model['detector_weights']).name}"
    if model.get("encoder_weights"):
        model["encoder_weights"] = f"{MODELS_MOUNT}/{Path(model['encoder_weights']).name}"
    return raw


def _download_dir(remote_path: str, local_dir: Path) -> None:
    from modal.volume import FileEntryType

    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in _outputs_vol.iterdir(remote_path, recursive=False):
        name = Path(entry.path).name
        target = local_dir / name
        if entry.type == FileEntryType.DIRECTORY:
            _download_dir(entry.path, target)
        else:
            with target.open("wb") as f:
                for chunk in _outputs_vol.read_file(entry.path):
                    f.write(chunk)
