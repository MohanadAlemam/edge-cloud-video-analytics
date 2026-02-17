#-----------------------------------------------------------------------------------------------------------------------
# CLOUD INFERENCE SERVER LOGIC
#-----------------------------------------------------------------------------------------------------------------------

"""
Cloud Inference Server
=====================
Flask server for YOLO object detection inference.
Provides /inference and /metrics endpoints.
"""

from flask import Flask, request, jsonify
import time
import argparse

from .utilities import _load_model, _read_frame_from_request

from common.visualize import parse_detections
from common.frame_content import detect_intrusion

app = Flask(__name__)

# ENDPOINT : When someone sends a request to infer, run this
@app.route('/infer', methods=['POST'])

def inference():
    """
    Conduct inference using YOLO model. Receives requests from the edge.

    :return: the inference result including detections and time to process the frame.
    """
    frame_size = 640

    frame = _read_frame_from_request(request)

    if frame is None:
        return jsonify({"error": "No frame received."}), 400

    model = _load_model() # model name set in __main__
    if model is None:
        return jsonify({"processing_time_ms":0.0, "detections":[]}), 503 # Service unavailable

    start = time.time()

    try:
        results = model(frame, imgsz= frame_size, verbose=False) # produces ultralytics object
    except Exception as e:
        # if inference failure record zero ms, and return 500 error
        return jsonify({"processing_time_ms": 0.0, "detections":[], "error": "inference failed"}), 500 # internal error

    process_time = (time.time() - start) * 1000.0
    # ms time

    frame_result = results[0]
    detections = parse_detections(frame_result, model = model)

    intrusion_content = detect_intrusion(detections)
    # compute content

    return jsonify({"processing_time_ms":float(process_time),
                    "detections":detections,
                    "intrusion_metrics": intrusion_content}), 200 # 200 = success

# CLI
if __name__ == '__main__':
    # Commandline argument parsing

    parser = argparse.ArgumentParser(
        description='Cloud inference: cloud model inference process.'
    )

    # Model path and its default value
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/yolov8s.pt",
        help="Path to cloud YOLO model"
    )
    parser.add_argument(
        "--port", type=int,
        default=5000,
        help="Port number for the cloud server")

    args = parser.parse_args()

    # set the global model path in cloud.utilities
    import cloud
    cloud.utilities._MODEL_PATH = args.model_path
    _load_model(args.model_path)
    # preload the model for a warm start

    app.run(host='0.0.0.0', port=args.port, debug=True)
    # run the app and listen to queries

    # edge -side : '--server_url" default = "http://10.0.0.3:5000",
