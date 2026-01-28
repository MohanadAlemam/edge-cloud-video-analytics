import argparse
# Commandline argument parsing

from flask import Flask, request, jsonify
import cv2
import numpy as np
import time


try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


app = Flask(__name__)


_cloud_model = None
_cloud_model_loaded = False

cloud_metrics = {"total_requests_handled": 0,
                 "per_frame_processing_ms": []}

def _load_model(model_name:str = "yolov8s.pt"):
# yolov8s.pt as default is it faster and have good accuracy
    """
    Lazy load the chosen YOLO model on first request, or at startup if called.
    Returns model or None.

    :return: the model or None.
    """
    global _cloud_model_loaded, _cloud_model # modify the global variables
    # Ensures the model is loaded once and reused across all inference requests

    # Try to load YOLO from Ultralytics
    if _cloud_model_loaded:
        return _cloud_model

    if not _HAS_YOLO:
        print("Ultralytics YOLO is not installed in this system.")
        return None
    # Try to load the specified model from YOLO
    try:
        global _cloud_model
        _cloud_model = YOLO(model_name) # load the chosen model
        _cloud_model_loaded = True
        print(f"Cloud model loaded: {model_name}")
        return _cloud_model

    except Exception:
        print(f"Cloud Model: {model_name} failed to load from YOLO.")
        _cloud_model_loaded = False
        _cloud_model = None
        return None

def _read_frame_from_request(request):
    """
    Reads frame from a request payload, and decode the bytes into colored frame.

    :param request: request payload
    :return: colored frame, or none
    """
    if "image" not in request.files:
        # JPEG as a multipart file/payload to server/infer produced by .post()
        return None

    raw_bytes = request.files["image"].read() # read the raw bytes
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8) # convert to np array
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # decode to jpeg with RGB channels
    return frame


def parse_detections(result):
    """
    Convert single ultralytics result object to a list of dicts:
    with class name, confidence, bounding box for each object detected in the frame.

    :param result: ultralytics result object
    :return: list of dicts of objects detected in the frame
    """

    frame_detections = []

    try:
        boxes = getattr(result, "boxes", []) # access the ultralytics object

        for box in boxes:
            # each box represents a detected item, with:
            # box.cls= class index, box.conf = confidence score, box.xyxy = bounding box coordinates
            confidence = float(box.conf, 0.0) if hasattr(box, "conf") else 0.0
            class_index = int(box.cls) if hasattr(box, "cls") else -1
            # try to get the class index if no index map it to -1 =unknown

            if _cloud_model and hasattr(_cloud_model, "names"): # ensure we have the model , with attribute 'names'
            # try to map the classes index to class name
                class_name = _cloud_model.names.get(class_index, str(class_index))
            else:
                class_name = str(class_index)

            bounding_box = []
            # Initialize an empty list for the bounding box
            if hasattr(box, "xyxy"):
            # Check if the detection object actually has bounding-box coordinates
            # YOLO stores bounding boxes in "xyxy" format
                try:
                    coordinates = box.xyxy[0].cpu().numpy().tolist()
                        # box.xyxy is usually a PyTorch tensor. [0]: get the first and only box for this detection
                        # convert to cpu as PyTorch tensors live on the GPU and cannot be directly converted to NumPy.
                        # NumPy only works with CPU memory
                    bounding_box = [float(x) for x in coordinates]
                        # convert all coords to floats eg [x1, y1, x2, y2] all floats
                except Exception:
                    bounding_box = []

            frame_detections.append({"class": class_name,
                                     "confidence": confidence,
                                     "bounding_box": bounding_box
                                     })
    except Exception:
        return []
    return frame_detections # a list of dicts







