from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ultralytics.engine.results import Results

from tabletop_vision.learned.models import (
    SegmentationPrediction
)

class InstanceSegmenter:
    """Run learned instance segmentation on camera frames."""

    def __init__(
            self,
            model_path: str | Path,
            confidence_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must lie in [0, 1]."
            )

        self._model = YOLO(
            str(model_path)
        )

        self._confidence_threshold = (
            confidence_threshold
        )

    def predict(
            self,
            frame: np.ndarray,
    ) -> list[SegmentationPrediction]:

        #model supports either returning an iterator or list
        #we enforce list as our expected output for type safety
        results = list(
            self._model.predict(
            source=frame,
            conf=self._confidence_threshold,
            verbose=False,
            stream=False,
            )
        )

        if not results: 
            return []

        result = results[0]

        if not isinstance(result, Results):
            raise RuntimeError(
                "Segmentation model returned an unexpected result type."
            )

        if (
            result.masks is None
            or result.boxes is None
        ):
            return []

        predictions: list[
            SegmentationPrediction
            ] = []

        #We utilise masks.xy because it gives polygon coordinates
        #directly in pixels
        polygons = result.masks.xy
        masks = result.masks.data

        for index in range(len(result.boxes)):
            class_id = int(
                result.boxes.cls[index].item()
            )

            confidence = float(
                result.boxes.conf[index].item()
            )

            class_name = str(
                result.names[class_id]
            )

            mask = (
                masks[index]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            polygon = np.asarray(
                polygons[index],
                dtype=np.float32,
            )

            predictions.append(
                SegmentationPrediction(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    mask=mask,
                    polygon=polygon,
                )
            )

        return predictions


