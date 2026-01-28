from logging import exception

from flask import Flask, request, jsonify
import cv2
import numpy as np
import time
import threading



try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


app = Flask(__name__)


_cloud_model = None
_cloud_model_loaded = False

_requests_handled = 0
_inference_time = []


def _load_model(model_name:str):
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
        cloud_model = YOLO(model_name) # load the chosen model
        cloud_model_loaded = True
        print(f"YOLO model loaded: {model_name}")
        return cloud_model
    except Exception:
        print(f"Mode {model_name} failed to load from YOLO.")
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






