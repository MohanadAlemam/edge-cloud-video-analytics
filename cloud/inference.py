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






