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

        # A small size to run a dummy inference, this warmup speeds up later calls (pto reduces end to end time)
        self.warmup_size = warmup_size


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





