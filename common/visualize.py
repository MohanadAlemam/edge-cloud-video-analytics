########################################################################################################################
# COMMON PROCESSES
########################################################################################################################

import cv2
import numpy as np

def parse_detections(frame_result: list[dict], model):
    """

    Convert a single ultralytics result object to a list of detection dicts:
    [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]

    :param frame_result: ultralytics result object
    :param model: model
    :return: list of dicts of objects detected in the frame
    """

    frame_detections = []

    try:
        boxes = getattr(frame_result, "boxes", []) # access the ultralytics object

        for box in boxes:
            # each box represents a detected item, with:
            # box.cls= class index - box.conf = confidence score - box.xyxy = bounding box coordinates
            confidence = float(getattr(box, "conf", 0.0))
            class_index = int(box.cls) if hasattr(box, "cls") else -1
            # try to get the class index if no index map it to -1 =unknown

            if model and hasattr(model, "names"): # ensure we have the model with attribute 'names'
            # try to map the classes index to class name
                class_name = model.names.get(class_index, str(class_index))
            else:
                class_name = str(class_index)

            bounding_box = []
            # Initialize an empty list for the bounding box
            if hasattr(box, "xyxy"):
            # make sure object actually has bounding box coordinates
            # YOLO stores bounding boxes in "xyxy" format
                try:
                    coordinates = box.xyxy[0].cpu().numpy().tolist()
                        # box.xyxy is usually a PyTorch tensor. [0]: get the first and only box for this detection
                        # convert to cpu pytorch tensors live on the GPU and cannot be directly converted to NumPy
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


# Function to draw detections on frames
def annotate_frame(resized_frame: np.ndarray, model_response: dict):
    """
    Draw bounding boxes, and shows labels on each frame based on the cloud_response.

    :param resized_frame: frame to draw on, the small/resized frame
    :param model_response: dict returned from cloud inference which contains 'detections'
    :return: annotated frame which can be written to video or displayed
    """

    detections = model_response.get("detections", [])
    display_frame = resized_frame.copy()  # copy of the resized frame

    # Unpack detection object
    for detection in detections:
        cls = detection.get("class", "unknown")
        # get the class, if doesnt exit use a generic label "unknown"
        confidence = detection.get("confidence", 0.0)
        bounding_box = detection.get("bounding_box", [])

        if len(bounding_box) == 4:
            x1, y1, x2, y2 = map(int, bounding_box)
            # converts float values to integers, OpenCV requires ints

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # (0, 255, 0) green, and 2 is the thickness of the borders

            cv2.putText(
                display_frame,
                f"{cls}: {confidence:.2f}",
                (x1, max(0, y1 - 8)),  # put the text slightly above the bounding box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # font size
                (255, 255, 255),  # color of the font
                1,
            )
    return display_frame
