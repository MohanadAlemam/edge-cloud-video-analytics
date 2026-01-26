# a lightweight edge model to reduce traffic to the cloud

import numpy as np
import threading # to do non-blocking background warmup



class EdgeModel:
    def __init__(self,
                 model_name: str="yolov8n.pt",
                 min_confidence: float=0.3,
                 warmup_size: int=100):


        self.model_name = model_name
        self.min_confidence = min_confidence

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

