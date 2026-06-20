"""
Estrategia de conteo por cruce de línea con tiling.

Toma el cuadrado central del frame, recorta la franja central (ancho = mitad
del lado) y la divide en dos tiles cuadrados apilados: superior e inferior.
Cada tile se procesa con su propia instancia de YOLO, así el tracking de un
tile es independiente del otro. Cada tile cuenta cruces de una línea vertical
en su centro con cruce estricto, y el total es la suma de ambos.

El tiling busca reducir el churn de track_id: cada tile tiene la mitad de
objetos y, al reescalarse a la resolución de inferencia, el arándano se ve más
grande. La frontera horizontal entre tiles es paralela al movimiento, así que
un arándano vive en un solo tile, salvo los que caen justo sobre el corte.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from ops.strategies.line_crossing import (
    ObjectCounter,
    _annotate_boxes,
    _crop_center_square,
    _device,
    _to_tracking_data,
)


def count_video(
    video_path: Path,
    detector_weights: Path,
    params: dict,
    device: str,
    render_path: Path | None = None,
) -> int:
    """Cuenta cruces de línea sobre la franja central partida en 2 tiles.

    Cada tile (superior e inferior) usa una instancia YOLO propia con tracker
    independiente y su propio ObjectCounter sobre una línea vertical en el
    centro del tile. El resultado es la suma de los dos tiles.

    Args:
        video_path: ruta al archivo .mp4.
        detector_weights: ruta a los pesos YOLO (.pt).
        params: dict con direction y conf_threshold.
        device: "gpu" | "cpu".
        render_path: si se da, escribe un .mp4 con los 2 tiles anotados
            apilados verticalmente.

    Returns:
        Total de objetos contados, sumando ambos tiles.
    """
    from ultralytics import YOLO

    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    if not detector_weights.exists():
        raise FileNotFoundError(f"Pesos del detector no encontrados: {detector_weights}")

    model_top = YOLO(str(detector_weights))
    model_bottom = YOLO(str(detector_weights))
    counter_top = ObjectCounter("horizontal", 0.5, params["direction"])
    counter_bottom = ObjectCounter("horizontal", 0.5, params["direction"])

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tile_top, tile_bottom = _center_strip_tiles(frame)
            result_top = model_top.track(
                tile_top,
                persist=True,
                conf=params["conf_threshold"],
                device=_device(device),
                verbose=False,
            )[0]
            result_bottom = model_bottom.track(
                tile_bottom,
                persist=True,
                conf=params["conf_threshold"],
                device=_device(device),
                verbose=False,
            )[0]
            counter_top.update(_to_tracking_data(result_top, 1, 0, 1))
            counter_bottom.update(_to_tracking_data(result_bottom, 1, 0, 1))

            if render_path is not None:
                annotated = _draw_tiles(result_top, result_bottom, counter_top, counter_bottom)
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(render_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                writer.write(annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[tiled_crossing] video anotado escrito en {render_path}")

    return counter_top.get_total_crossed() + counter_bottom.get_total_crossed()


def _center_strip_tiles(frame):
    """Recorta el cuadrado central, toma la franja central de ancho = mitad del
    lado y la divide en tile superior e inferior, ambos cuadrados.

    Devuelve (tile_top, tile_bottom). El corte horizontal entre tiles está a
    media altura; la línea de conteo es el centro vertical de cada tile.
    """
    view, _offset_x, side = _crop_center_square(frame)
    half = side // 2
    strip_x0 = (side - half) // 2
    strip = view[:, strip_x0:strip_x0 + half]
    return strip[:half, :], strip[half:half * 2, :]


def _draw_tiles(
    result_top,
    result_bottom,
    counter_top: ObjectCounter,
    counter_bottom: ObjectCounter,
):
    """Apila los 2 tiles anotados en un solo frame para el render.

    Cada tile lleva las cajas de result.plot() (las contadas en rojo), el track
    id centrado, la línea vertical de conteo y su subtotal; abajo va el total.
    """
    total = counter_top.get_total_crossed() + counter_bottom.get_total_crossed()
    tiles = []
    for result, counter in ((result_top, counter_top), (result_bottom, counter_bottom)):
        f = result.plot(conf=False, labels=False, line_width=1)
        height, width = f.shape[:2]
        _annotate_boxes(f, result, counter)
        x = width // 2
        cv2.line(f, (x, 0), (x, height), (0, 0, 255), 2)
        cv2.putText(
            f,
            f"tile: {counter.get_total_crossed()}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(f)
    stacked = cv2.vconcat(tiles)
    cv2.putText(
        stacked,
        f"total: {total}",
        (10, stacked.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return stacked
