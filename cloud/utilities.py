import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


_cloud_model = None
_cloud_model_loaded = False
_MODEL_PATH = None

def _load_model(model_path:str = None):
# yolov8.pt as default is it faster and have good accuracy
    """
    Lazy load the chosen YOLO model on first request, or at startup if called.
    Returns model or None.

    :return: the model or None.
    """
    global _cloud_model_loaded, _cloud_model, _MODEL_PATH # modify the global variables
    # Ensures the model is loaded once and reused across all inference requests

    if model_path is None:
        model_path = _MODEL_PATH # sent model path to the global Model path

    if _cloud_model_loaded and _cloud_model is not None:
        return _cloud_model

    if not _HAS_YOLO:
        print("Ultralytics YOLO is not installed in this system.")
        return None
    # Try to load the specified model from YOLO
    try:
        _cloud_model = YOLO(model_path) # load the chosen model
        _cloud_model_loaded = True
        print(f"Cloud model loaded: {model_path}")
        return _cloud_model

    except Exception:
        print(f"Cloud Model: {model_path} failed to load from YOLO.")
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

