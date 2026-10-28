"""Camada de inferência do modelo YOLO usado pela aplicação."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"


@dataclass(frozen=True)
class DetectionResult:
    """Resultado simplificado para consumo pela interface."""

    annotated_image: Image.Image
    confidences: tuple[float, ...]

    @property
    def has_acne(self) -> bool:
        return bool(self.confidences)

    @property
    def count(self) -> int:
        return len(self.confidences)

    @property
    def max_confidence(self) -> float | None:
        return max(self.confidences, default=None)


class AcneDetector:
    """Carrega o checkpoint uma vez e executa detecção em imagens PIL."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {self.model_path.resolve()}"
            )
        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.15,
        iou: float = 0.70,
    ) -> DetectionResult:
        """Detecta lesões de acne e devolve a imagem com as marcações."""

        if not 0.0 < confidence <= 1.0:
            raise ValueError("O limiar de confiança deve estar entre 0 e 1.")
        if not 0.0 < iou <= 1.0:
            raise ValueError("O limiar de IoU deve estar entre 0 e 1.")

        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        prediction = self.model.predict(
            source=np.asarray(rgb_image),
            conf=confidence,
            iou=iou,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]

        boxes = prediction.boxes
        confidences = (
            tuple(float(value) for value in boxes.conf.detach().cpu().tolist())
            if boxes is not None
            else ()
        )

        # O Ultralytics desenha em BGR; o PIL/Streamlit espera RGB.
        annotated_bgr = prediction.plot(labels=True, conf=True)
        annotated_rgb = np.ascontiguousarray(annotated_bgr[..., ::-1])

        return DetectionResult(
            annotated_image=Image.fromarray(annotated_rgb),
            confidences=confidences,
        )

