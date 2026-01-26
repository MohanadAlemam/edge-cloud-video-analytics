# a lightweight edge model to reduce traffic to the cloud
import time
from os import name

import cv2
import numpy as np
import threading # to do non-blocking background warmup
from preprocess import convert_to_grayscale

from pygments.lexer import default


class EdgeModel:
    def __init__(self,
                 model_name: str="yolov8n.pt",
                 edge_confident_threshold: float=0.30,
                 warmup_size: int=100):


        self.model_name = model_name
        self.edge_confident_threshold = edge_confident_threshold

        # A small size to run a dummy inference, this warmup speeds up later calls (to reduces end to end time)
        self.warmup_size = warmup_size

        self._model_loaded = False
        self._use_fallback = False
        self._model = None

    # 1. import the backage
    def _import_yolo_package(self):
        """
        Import the YOLO class at runtime.
        This defers the ultralytics import until the first time the edge model is actually used.

        """

        try:
            from ultralytics import YOLO
            # local import to avoid heavy import at module load time
            return YOLO

        except Exception as e:
            print(f"Error: loading Edge Model 'YOLO': {e}")
            return None

    # 2. Try to conduct a background warmup to avoid blocking the pipeline
    def _background_warmup(self):
        try:
            dummy = np.zeros((self.warmup_size, self.warmup_size, 3), dtype=np.uint8)
            _ = self._model(dummy, imgsz=self.warmup_size, verbose=False)

        except Exception:
            pass # ignore warmup errors

    # 3. Load the model if not loaded and do the warmup once.
    def _load_edge_model(self, background_warmup = True):
        if self._model_loaded or self._use_fallback:
            # If already loaded or fallback decided, do nothing
            return

        YOLO = self._import_yolo_package()

        if YOLO is None:
                # if the import function returns None
            print(f"Error: loading Edge Model, EdgeModel not available; using fallback.")
            self._use_fallback = True # Activate fallback logic
            return

        try:
            self._model = YOLO(self.model_name)
            # download model's weights file if not already downloaded
            self._model_loaded = True

            if background_warmup:
            # start warmup in background so we don't block the pipeline
                thread = threading.Thread(target=self._background_warmup, daemon=True)
            # imgsz=self.warmup_size : resize dummy frame to a small square to do cheap warmup initialization.

                thread.start()

            else:
                self._background_warmup()
                    # synchronous the warmup blocking the pipline
        except Exception as e:
            print(f"Error: loading Edge Model, EdgeModel not available; using fallback.")
            self._use_fallback = True

    def _detect_edge(self, colored_frame: np.ndarray):
        """
        Make lightweight prediction on the frame. Returns a list of confidence scores.
        Empty list means 'not interesting'.

        :param colored_frame: frame
        :return: confidence list
        """

        if colored_frame is None:
            return []
        # If caller passed nothing false return "not interesting"

        self._load_edge_model()
        # load the model

        # If YOLO unavailable, fall back to cheap detector
        if self._use_fallback or not self._model_loaded:
            # in case edge model fails to load, or we have activated fallback protocol
            decision, confidence, _ = self._fallback_contour_detector(colored_frame)
            if decision:
                return [confidence]
            else:
                return []

        # Else run the Yolo inference
        try:
            results = self._model(colored_frame, imgsz=416, verbose=False)
            # Run the model on the image, ultralytics accepts np images.
            result_for_image = results[0]

            # Get boxes object safely.
            boxes_container = getattr(result_for_image, "boxes", [])

            # Extract confidence scores only
            confidences = [float(box.conf) for box in boxes_container]
            # list comprehension to get the list of confidence for the objects detected in the frame
            return confidences
        except Exception:
            # in case of failure activate fallback protocol
            decision, confidence, _ = self._fallback_contour_detector(colored_frame)
            if decision:
                return [confidence]
            else:
                return []






