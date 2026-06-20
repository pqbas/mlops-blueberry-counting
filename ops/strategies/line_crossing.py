"""
Estrategia de conteo por cruce de línea.

Corre YOLO con tracking sobre el video, normaliza los centroides y los pasa a
un ObjectCounter que cuenta los objetos que cruzan una línea virtual.

El ObjectCounter aplica cruce estricto: un track se cuenta una sola vez, cuando
su centroide transita del lado de origen al lado opuesto entre dos updates
consecutivos. Coincide con el GT de groundtruth.count_line_crossing.
"""
from __future__ import annotations

from pathlib import Path

import cv2


class ObjectCounter:
    """Conteo neto por cruce de línea sobre datos de tracking en coordenadas
    normalizadas [0, 1].

    Cada track_id contribuye según su último cruce: cruzar del lado de origen
    al opuesto lo agrega al set; cruzar de regreso lo quita. El total es
    len(crossed), o sea los tracks que ahora mismo están del lado opuesto. Así
    un arándano que oscila o que se re-identifica al volver no infla el conteo.
    Coincide con el GT de groundtruth.count_line_crossing.
    """

    def __init__(self, count_mode: str, threshold: float, direction: str):
        self.count_mode = count_mode
        self.threshold = threshold
        self.direction = direction

        self._prev: dict[int, float] = {}
        self.crossed: set[int] = set()

        if direction in ("top2down", "left2right"):
            self.count_condition = lambda c: c > threshold
        elif direction in ("down2top", "right2left"):
            self.count_condition = lambda c: c < threshold
        else:
            raise ValueError(f"Dirección inválida: {direction}")

    def update(self, tracking_data: list[dict]) -> None:
        for obj in tracking_data:
            track_id = obj["track_id"]
            coord = obj["cx"] if self.count_mode == "horizontal" else obj["cy"]
            prev = self._prev.get(track_id)
            if prev is not None:
                if not self.count_condition(prev) and self.count_condition(coord):
                    self.crossed.add(track_id)  # cruce hacia adelante
                elif self.count_condition(prev) and not self.count_condition(coord):
                    self.crossed.discard(track_id)  # cruce de regreso
            self._prev[track_id] = coord

    def get_total_crossed(self) -> int:
        """Total neto: tracks cuyo último cruce los dejó del lado opuesto."""
        return len(self.crossed)

    def reset(self) -> None:
        self._prev.clear()
        self.crossed.clear()


def count_video(
    video_path: Path,
    detector_weights: Path,
    params: dict,
    device: str,
    render_path: Path | None = None,
) -> int:
    """Cuenta objetos que cruzan la línea en un video.

    Lee el video frame a frame con cv2; si params["crop"] == "center_square"
    recorta un cuadrado centrado (lado = alto del frame) antes de pasar el
    frame a YOLO. Las cajas detectadas se remapean a coordenadas normalizadas
    del frame completo, así el contador y el ground truth comparten la línea.

    Args:
        video_path: ruta al archivo .mp4.
        detector_weights: ruta a los pesos YOLO (.pt).
        params: dict con count_mode, threshold, direction, conf_threshold, crop.
        device: "gpu" | "cpu".
        render_path: si se da, escribe en esa ruta un .mp4 anotado con cajas,
            track ids, la línea de conteo y el conteo acumulado.

    Returns:
        Total de objetos contados que cruzaron la línea.
    """
    from ultralytics import YOLO

    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    if not detector_weights.exists():
        raise FileNotFoundError(f"Pesos del detector no encontrados: {detector_weights}")

    model = YOLO(str(detector_weights))
    counter = ObjectCounter(params["count_mode"], params["threshold"], params["direction"])
    crop = params["crop"]

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
            frame_width = frame.shape[1]
            if crop == "center_square":
                view, offset_x, side = _crop_center_square(frame)
            else:
                view, offset_x, side = frame, 0, frame_width

            result = model.track(
                view,
                persist=True,
                conf=params["conf_threshold"],
                device=_device(device),
                verbose=False,
            )[0]
            counter.update(_to_tracking_data(result, frame_width, offset_x, side))

            if render_path is not None:
                annotated = _draw_frame(result, counter, params, frame_width, offset_x)
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
            print(f"[line_crossing] video anotado escrito en {render_path}")

    return counter.get_total_crossed()


def _crop_center_square(frame):
    """Recorta un cuadrado centrado horizontalmente, de lado = alto del frame.

    Devuelve (crop, offset_x, side). En video apaisado el crop conserva el alto
    completo; offset_x es el desplazamiento en píxeles del borde izquierdo.
    """
    height, width = frame.shape[:2]
    side = min(height, width)
    offset_x = (width - side) // 2
    return frame[:, offset_x:offset_x + side], offset_x, side


def _draw_frame(result, counter: ObjectCounter, params: dict, frame_width: int, offset_x: int):
    """Frame anotado sobre el view (recortado si hay crop): cajas via
    result.plot(), las contadas se repintan de rojo y el track id va centrado,
    más la línea de conteo y el contador acumulado."""
    frame = result.plot(conf=False, labels=False, line_width=1)
    height, view_width = frame.shape[:2]
    _annotate_boxes(frame, result, counter)
    if params["count_mode"] == "vertical":
        y = int(params["threshold"] * height)
        cv2.line(frame, (0, y), (view_width, y), (0, 0, 255), 2)
    else:
        x = int(params["threshold"] * frame_width) - offset_x
        cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"count: {counter.get_total_crossed()}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def _annotate_boxes(frame, result, counter: ObjectCounter) -> None:
    """Repinta de rojo las cajas ya contadas y escribe el track id centrado.

    Dibuja sobre el frame ya anotado por result.plot(); las coordenadas xyxy
    están en píxeles del view, igual que el frame, así que se usan directo.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return
    counted = counter.crossed
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    for (x1, y1, x2, y2), track_id in zip(xyxy, ids):
        tid = int(track_id)
        if tid in counted:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        _put_centered_id(frame, tid, (x1 + x2) / 2, (y1 + y2) / 2)


def _put_centered_id(frame, track_id: int, cx: float, cy: float) -> None:
    """Escribe el track id en texto blanco centrado en (cx, cy)."""
    text = str(track_id)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(cx - text_w / 2), int(cy + text_h / 2))
    cv2.putText(frame, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _to_tracking_data(result, frame_width: int, offset_x: int, side: int) -> list[dict]:
    """Convierte un resultado de YOLO.track() en dicts con track_id y centroide
    normalizado al frame completo.

    YOLO ve el view (posiblemente recortado): boxes.xywhn está normalizado a ese
    view. cx se remapea al frame completo; cy no cambia porque el crop conserva
    el alto. Sin crop (offset_x=0, side=frame_width) el remapeo es identidad.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn.cpu().numpy()
    ids = boxes.id.int().cpu().numpy()
    return [
        {
            "track_id": int(track_id),
            "cx": (offset_x + float(cx) * side) / frame_width,
            "cy": float(cy),
        }
        for (cx, cy, _w, _h), track_id in zip(xywhn, ids)
    ]


def _device(device: str) -> str | int:
    return 0 if device == "gpu" else "cpu"
