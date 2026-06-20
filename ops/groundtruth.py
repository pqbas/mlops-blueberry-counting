"""
Ground truth de conteo a partir de labels con track_id.

Cada estrategia tiene su GT natural, porque cuentan cantidades distintas:

  - line_crossing -> cruce estricto de línea: un track se cuenta cuando su
    centroide transita del lado de origen al lado opuesto entre dos frames
    anotados consecutivos. Port de gt_counter.py del backend.

  - embeddings -> membresía en región: un track se cuenta cuando aparece en
    al menos `min_track_length` frames con su centroide dentro de la región.

Las coordenadas de los labels son normalizadas [0, 1], igual que el threshold
y la región.
"""
from __future__ import annotations

from pathlib import Path

from ops.config import ConfigError


def _frame_index(txt: Path) -> int:
    """Índice de frame desde el nombre del archivo, acepta `frame_NNNNNN.txt`
    o `NNNNNN.txt`."""
    return int(txt.stem.removeprefix("frame_"))


def load_track_labels(labels_dir: Path) -> list[tuple[int, list[dict]]]:
    """Lee los labels por frame de un video, ordenados por frame_idx.

    Cada archivo `frame_NNNNNN.txt` (o `<frame_idx>.txt`) contiene una línea
    por objeto con el formato YOLO extendido: `class cx cy w h track_id`.

    Retorna [(frame_idx, [label, ...]), ...] donde cada label es un dict con
    class_id, cx, cy, w, h y track_id.
    """
    if not labels_dir.is_dir():
        raise ConfigError(f"Directorio de labels no encontrado: {labels_dir}")

    frames: list[tuple[int, list[dict]]] = []
    for txt in sorted(labels_dir.glob("*.txt"), key=_frame_index):
        labels: list[dict] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            cls, cx, cy, w, h, track_id = parts[:6]
            labels.append(
                {
                    "class_id": int(cls),
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "track_id": int(track_id),
                }
            )
        frames.append((_frame_index(txt), labels))
    return frames


def _make_condition(direction: str):
    if direction in ("top2down", "left2right"):
        return lambda coord, threshold: coord > threshold
    if direction in ("down2top", "right2left"):
        return lambda coord, threshold: coord < threshold
    raise ValueError(f"Dirección inválida: {direction}")


def count_line_crossing(
    frames_labels: list[tuple[int, list[dict]]],
    count_mode: str,
    threshold: float,
    direction: str,
    class_filter: int | None = None,
) -> int:
    """GT por conteo neto de cruce de línea: cada track_id se agrega al cruzar
    hacia adelante y se quita al cruzar de regreso; el total es len(crossed)."""
    cond = _make_condition(direction)
    use_x = count_mode == "horizontal"

    prev_coord: dict[int, float] = {}
    crossed: set[int] = set()

    for _frame_idx, labels in frames_labels:
        for label in labels:
            if class_filter is not None and label["class_id"] != class_filter:
                continue
            track_id = label["track_id"]
            coord = label["cx"] if use_x else label["cy"]
            prev = prev_coord.get(track_id)
            if prev is not None:
                if not cond(prev, threshold) and cond(coord, threshold):
                    crossed.add(track_id)
                elif cond(prev, threshold) and not cond(coord, threshold):
                    crossed.discard(track_id)
            prev_coord[track_id] = coord

    return len(crossed)


def count_region_membership(
    frames_labels: list[tuple[int, list[dict]]],
    region: tuple[float, float, float, float],
    min_track_length: int = 1,
    class_filter: int | None = None,
) -> int:
    """GT por membresía en región: track_ids únicos vistos dentro de la región
    en al menos `min_track_length` frames."""
    x1, y1, x2, y2 = region
    appearances: dict[int, int] = {}

    for _frame_idx, labels in frames_labels:
        for label in labels:
            if class_filter is not None and label["class_id"] != class_filter:
                continue
            if x1 <= label["cx"] <= x2 and y1 <= label["cy"] <= y2:
                track_id = label["track_id"]
                appearances[track_id] = appearances.get(track_id, 0) + 1

    return sum(1 for count in appearances.values() if count >= min_track_length)
