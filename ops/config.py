from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import yaml

Runtime = Literal["localhost", "modal"]
Device = Literal["gpu", "cpu"]
OpType = Literal["count", "evaluate"]
Strategy = Literal["line_crossing", "embeddings"]

_VALID_RUNTIMES = {"localhost", "modal"}
_VALID_DEVICES = {"gpu", "cpu"}
_VALID_TYPES = {"count", "evaluate"}
_VALID_STRATEGIES = {"line_crossing", "tiled_crossing", "embeddings"}
_VALID_COUNT_MODES = {"vertical", "horizontal"}
_VALID_DIRECTIONS = {"top2down", "down2top", "left2right", "right2left"}
_VALID_CROPS = {"none", "center_square"}

# count_mode usa el eje y -> direcciones verticales; horizontal usa el eje x.
_DIRECTIONS_BY_MODE = {
    "vertical": {"top2down", "down2top"},
    "horizontal": {"left2right", "right2left"},
}

# Campos de 'params' que aceptan varios valores separados por coma (sweep).
# 'region' queda fuera a propósito: su propio valor ya es una lista x1,y1,x2,y2.
_SWEEP_KEYS = {
    "line_crossing": ["count_mode", "threshold", "direction", "conf_threshold", "crop"],
    "tiled_crossing": ["direction", "conf_threshold"],
    "embeddings": [
        "conf_threshold",
        "crop_padding",
        "similarity_threshold",
        "max_staleness",
        "min_track_length",
    ],
}


@dataclass(frozen=True)
class Config:
    name: str
    type: OpType
    strategy: Strategy
    runtime: Runtime
    device: Device
    render: bool  # si True, escribe un .mp4 anotado por video
    dataset: Path  # ruta al manifest.yaml del dataset
    detector_weights: Path
    encoder_weights: Path | None
    params: dict  # params validados y tipados, específicos de la estrategia
    raw: dict


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------
# Carga y expansión de sweeps
# --------------------------------------------------------------------------
def load_config(path: Path) -> Config:
    return parse_dict(load_yaml(path))


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"YAML no encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"YAML inválido en {path}: se esperaba un mapping en el nivel raíz")
    return data


def expand_configs(data: dict) -> list[tuple[Config, str | None]]:
    """Producto cartesiano de los ejes de sweep (model.detector_weights y los
    params escalares de la estrategia) con valores separados por coma.

    Retorna [(cfg, suffix), ...] donde suffix es None si hay un único trial,
    o '<idx>_<key>-<val>_...' nombrando los ejes que varían en el sweep.
    """
    strategy = data.get("strategy")
    if strategy not in _SWEEP_KEYS:
        # Estrategia inválida: parse_dict emite el error con el mensaje correcto.
        return [(parse_dict(data), None)]

    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError("Campo 'params' debe ser un mapping")
    model = data.get("model")

    # Ejes de sweep: model.detector_weights y los params escalares.
    present_keys: list[str] = []
    value_lists: list[list] = []
    if isinstance(model, dict) and "detector_weights" in model:
        weights = _split_scalar(model["detector_weights"])
        if len(weights) > 1:
            present_keys.append("detector_weights")
            value_lists.append(weights)
    for key in _SWEEP_KEYS[strategy]:
        if key not in params:
            continue
        present_keys.append(key)
        value_lists.append(_split_scalar(params[key]))

    if not value_lists:
        return [(parse_dict(data), None)]

    combos = list(product(*value_lists))
    multi = len(combos) > 1
    varying = {present_keys[i] for i, vals in enumerate(value_lists) if len(vals) > 1}

    results: list[tuple[Config, str | None]] = []
    for idx, combo in enumerate(combos):
        variant = copy.deepcopy(data)
        variant.setdefault("params", {})
        for key, value in zip(present_keys, combo):
            if key == "detector_weights":
                variant["model"]["detector_weights"] = value
            else:
                variant["params"][key] = value
        if multi:
            parts = [str(idx)]
            for key in present_keys:
                if key not in varying:
                    continue
                if key == "detector_weights":
                    parts.append(f"{key}-{Path(variant['model']['detector_weights']).stem}")
                else:
                    parts.append(f"{key}-{variant['params'][key]}")
            suffix: str | None = "_".join(parts)
        else:
            suffix = None
        results.append((parse_dict(variant), suffix))
    return results


def _split_scalar(value) -> list:
    """Divide un valor en una lista de candidatos de sweep.

    Las cadenas se parten por coma; cualquier otro tipo se devuelve tal cual.
    """
    if isinstance(value, str):
        return [piece.strip() for piece in value.split(",") if piece.strip()]
    return [value]


# --------------------------------------------------------------------------
# Parseo y validación
# --------------------------------------------------------------------------
def parse_dict(data: dict) -> Config:
    name = _require_str(data, "name")
    op_type = _require_choice(data, "type", _VALID_TYPES)
    strategy = _require_choice(data, "strategy", _VALID_STRATEGIES)
    runtime = _require_choice(data, "runtime", _VALID_RUNTIMES)
    device = _require_choice(data, "device", _VALID_DEVICES)
    render = _optional_bool(data, "render", default=False)
    dataset = Path(_require_str(data, "dataset"))

    model = _require_mapping(data, "model")
    detector_weights = Path(_require_str(model, "detector_weights"))
    encoder_weights: Path | None = None
    if strategy == "embeddings":
        encoder_weights = Path(_require_str(model, "encoder_weights"))

    params_raw = _require_mapping(data, "params")
    if strategy == "line_crossing":
        params = _parse_line_crossing_params(params_raw)
    elif strategy == "tiled_crossing":
        params = _parse_tiled_crossing_params(params_raw)
    else:
        params = _parse_embeddings_params(params_raw)

    return Config(
        name=name,
        type=op_type,  # type: ignore[arg-type]
        strategy=strategy,  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        render=render,
        dataset=dataset,
        detector_weights=detector_weights,
        encoder_weights=encoder_weights,
        params=params,
        raw=data,
    )


def _parse_line_crossing_params(p: dict) -> dict:
    count_mode = _as_choice(p, "count_mode", _VALID_COUNT_MODES)
    direction = _as_choice(p, "direction", _VALID_DIRECTIONS)
    if direction not in _DIRECTIONS_BY_MODE[count_mode]:
        raise ConfigError(
            f"direction='{direction}' es incompatible con count_mode='{count_mode}'. "
            f"Direcciones válidas: {sorted(_DIRECTIONS_BY_MODE[count_mode])}"
        )
    return {
        "count_mode": count_mode,
        "threshold": _as_unit_float(p, "threshold"),
        "direction": direction,
        "conf_threshold": _as_unit_float(p, "conf_threshold"),
        "crop": _as_optional_choice(p, "crop", _VALID_CROPS, "none"),
    }


def _parse_tiled_crossing_params(p: dict) -> dict:
    """direction horizontal y conf_threshold; la línea de conteo queda fija en
    el centro de cada tile, count_mode y threshold no son configurables."""
    return {
        "direction": _as_choice(p, "direction", {"left2right", "right2left"}),
        "conf_threshold": _as_unit_float(p, "conf_threshold"),
    }


def _parse_embeddings_params(p: dict) -> dict:
    return {
        "region": _as_region(p, "region"),
        "conf_threshold": _as_unit_float(p, "conf_threshold"),
        "crop_padding": _as_float(p, "crop_padding", min_value=0.0),
        "similarity_threshold": _as_unit_float(p, "similarity_threshold"),
        "max_staleness": _as_int(p, "max_staleness", min_value=0),
        "min_track_length": _as_int(p, "min_track_length", min_value=1),
    }


# --------------------------------------------------------------------------
# Helpers de coerción (toleran str porque el sweep parte los valores en texto)
# --------------------------------------------------------------------------
def _require_str(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"Campo '{key}' requerido como string no vacío")
    return value


def _optional_bool(data: dict, key: str, default: bool) -> bool:
    """Lee un booleano opcional del nivel raíz del YAML."""
    if key not in data or data[key] is None:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigError(f"Campo '{key}'='{value}' debe ser booleano (true | false)")
    return value


def _require_mapping(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Campo '{key}' requerido como mapping")
    return value


def _require_choice(data: dict, key: str, choices: set[str]) -> str:
    value = _require_str(data, key)
    if value not in choices:
        raise ConfigError(f"Campo '{key}'='{value}' inválido. Valores permitidos: {sorted(choices)}")
    return value


def _as_choice(params: dict, key: str, choices: set[str]) -> str:
    if key not in params:
        raise ConfigError(f"params.{key} es requerido")
    value = params[key]
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"params.{key}='{value}' inválido. Valores permitidos: {sorted(choices)}")
    return value


def _as_optional_choice(params: dict, key: str, choices: set[str], default: str) -> str:
    """Como _as_choice pero opcional: devuelve default si la key no está."""
    if key not in params or params[key] is None:
        return default
    value = params[key]
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"params.{key}='{value}' inválido. Valores permitidos: {sorted(choices)}")
    return value


def _as_float(params: dict, key: str, min_value: float | None = None, max_value: float | None = None) -> float:
    if key not in params:
        raise ConfigError(f"params.{key} es requerido")
    raw = params[key]
    if isinstance(raw, bool):
        raise ConfigError(f"params.{key} debe ser un número, no bool")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"params.{key}='{raw}' no es un número válido") from exc
    if min_value is not None and value < min_value:
        raise ConfigError(f"params.{key}={value} debe ser >= {min_value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"params.{key}={value} debe ser <= {max_value}")
    return value


def _as_unit_float(params: dict, key: str) -> float:
    return _as_float(params, key, min_value=0.0, max_value=1.0)


def _as_int(params: dict, key: str, min_value: int | None = None) -> int:
    if key not in params:
        raise ConfigError(f"params.{key} es requerido")
    raw = params[key]
    if isinstance(raw, bool):
        raise ConfigError(f"params.{key} debe ser un entero, no bool")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"params.{key}='{raw}' no es un entero válido") from exc
    if min_value is not None and value < min_value:
        raise ConfigError(f"params.{key}={value} debe ser >= {min_value}")
    return value


def _as_region(params: dict, key: str) -> tuple[float, float, float, float]:
    if key not in params:
        raise ConfigError(f"params.{key} es requerido (x1,y1,x2,y2 normalizado)")
    raw = params[key]
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    elif isinstance(raw, (list, tuple)):
        pieces = list(raw)
    else:
        raise ConfigError(f"params.{key} debe ser 'x1,y1,x2,y2' o una lista de 4 números")
    if len(pieces) != 4:
        raise ConfigError(f"params.{key} debe tener exactamente 4 valores, recibí {len(pieces)}")
    try:
        x1, y1, x2, y2 = (float(v) for v in pieces)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"params.{key}='{raw}' tiene valores no numéricos") from exc
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"params.{key}.{name}={value} fuera de rango [0, 1]")
    if x1 >= x2 or y1 >= y2:
        raise ConfigError(f"params.{key}: se requiere x1 < x2 e y1 < y2")
    return (x1, y1, x2, y2)
