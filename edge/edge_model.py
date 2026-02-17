import time
import numpy as np
import threading
# to do non-blocking background warmup

from common.visualize import parse_detections
from common.frame_content import detect_intrusion
# to count the number of vehicles and pedestrian

#-----------------------------------------------------------------------------------------------------------------------
# EDGE MODEL: LIGHTWEIGHT MODEL TO PERFORM ON-EDGE INFERENCE AND REDUCE TRAFFIC TO THE CLOUD
#-----------------------------------------------------------------------------------------------------------------------

class EdgeModel:
    """
    Edge Model class to create an instance of the edge model during runtime. It handles download of the backage.
    loading of the model weight inedge device and making decisions on whether to send the frame to the cloud or
    not based on the prediction confidence scores.

    - Edge Model: yolov8n.pt: yolo nano lightweight model, as a gateway to the cloud.
    """
    def __init__(self,
                 model_name: str="yolov8n.pt",
                 edge_conf_threshold: float=0.70,
                 warmup_size: int=100):


        self.model_name = model_name
        self.edge_conf_threshold = edge_conf_threshold

        # A small size to run a dummy inference, this warmup speeds up later calls
        self.warmup_size = warmup_size

        self._model_loaded = False
        self._model = None
        self._model_lock = threading.Lock()

    # 1. import the backage
    def _import_yolo_package(self):
        """
        Asynchronously Import ultralytics YOLO class at runtime.

        """
        try:
            from ultralytics import YOLO
            # local import to avoid heavy import at module load time
            return YOLO

        except Exception as e:
            print(f"Error: loading YOLO package: {e}")
            return None

    # 2. Try to conduct a background warmup to avoid blocking the pipeline
    def _background_warmup(self):
        """
        Asynchronously perform ultralytics YOLO warm up.

        :return: nothing
        """
        try:
            dummy = np.zeros((self.warmup_size, self.warmup_size, 3), dtype=np.uint8)
            #use the locked thread to warm-up
            with self._model_lock:
                _ = self._model(dummy, imgsz=self.warmup_size, verbose=False)

        except Exception:
            pass # ignore warmup errors

    # 3. Load the model if not loaded and do the warmup once.
    def _load_edge_model(self, background_warmup = True):
        if self._model_loaded:
            # If already loaded do nothing
            return

        YOLO = self._import_yolo_package()

        if YOLO is None:
                # if the import function returns None
            print(f"Error: loading YOLO package.")
            return

        try:
            self._model = YOLO(self.model_name)
            # download model's weights file if not already downloaded
            self._model_loaded = True

            if background_warmup:
            # start warmup in background so we don't block the pipeline
                thread = threading.Thread(target=self._background_warmup, daemon=True)
            # imgsz=self.warmup_size, resize dummy frame to a small square to do cheap warmup initialization.
                thread.start()

            else:
                self._background_warmup()
                    # synchronous the warmup blocking the pipline
        except Exception as e:
            print(f"Error: loading Edge Model ({self.model_name}). {e}")
            return

    def _detect_edge(self, colored_frame: np.ndarray):
        """
        Make lightweight prediction on the frame. Returns a list of confidence scores.
        Empty list means 'not interesting'.

        :param colored_frame: frame
        :return: confidence list
        """
        if colored_frame is None:
            return [], {}, 0.0 # no confidence lists, and elapse time is zero
        # If caller passed nothing false return "not interesting"

        self._load_edge_model()
        # load the model

        if not self._model_loaded:
                return [], {}, 0.0 # If YOLO unavailable,

        # Else run the Yolo inference
        try:
            start_time = time.time()
            # use the same thread
            with self._model_lock:
                results = self._model(colored_frame, imgsz=416, verbose=False)
            # Run the model on the image, ultralytics accepts np images.
            inference_ms = (time.time() - start_time) * 1000.0 # inference time for each frame ms

            result_for_image = results[0]
            detections = parse_detections(result_for_image, model=self._model)
            # detections = a list of dictionaries. keys "class" "confidence" "bounding_box"

            edge_response = {
                "processing_time_ms":float(inference_ms),
                "detections":detections,
            }

            # Extract confidence scores only
            detection_confidences = []
            for detection in detections:
                confidence = detection.get("confidence", 0.0)
                detection_confidences.append(confidence)

            return detection_confidences, edge_response, inference_ms
        except Exception:
                return [], {}, 0.0 # no confidence lists, and elapse time is zero

    def edg_model_decision(self, colored_frame: np.ndarray):
        """
        Decide whether to send the frame to cloud based on edge detections.

        Returns: (edge_decision: bool, confidences_list: List[float],
              edge_response: dict, inference_ms: float)
        """
        confidences_list, edge_response, inference_ms = self._detect_edge(colored_frame)

        default_intrusion = {
            "intrusion": False,
            "alert_level": "GREEN",
            "intrusion_content": {},
            "intrusion_count": 0
        }

        if not confidences_list:
            print("Edge model: Edge failed to detect objects. Sending to the cloud.")
            return True, [], edge_response, inference_ms
            # Send to could if not objects detected
            # Change to False, Prevent offloading (edge-only)

        send_to_cloud = max(confidences_list) < self.edge_conf_threshold

        if not send_to_cloud:
            intrusion_content = detect_intrusion(edge_response.get("detections", []))
            edge_response["intrusion_metrics"] = intrusion_content or default_intrusion

        return send_to_cloud, confidences_list, edge_response, inference_ms