import numpy as np

def detect_intrusion(detections: list[dict]) -> dict:
    """
    determine whether there is an intrusion or not based on count frame detections
    by class and return summary statistics. And produces alter level based on the classes
    of the objects detected on the frame.

    """
    # Define class groups
    critical_classes = {
        "person",  # any unauthorized person class 0
        "knife",  # Weapon
        "scissors",
        "baseball bat",
        # Can be used as weapon
    }

    amber_classes = {
        # Vehicles
        "car", "truck", "bus", "motorcycle", "bicycle",
        "boat", "train",
        # Large animals
        "dog", "horse", "cow", "sheep", "elephant", "bear",
        "zebra", "giraffe",
        # could contain threats
        "backpack", "handbag", "suitcase",  # 24, 26, 28
        # Aerial
        "airplane",
        "fork",
        # Unknown objects
        "unknown",
    }

    if not detections:
        return {"intrusion": False,
                "alert_level": "GREEN",
                "intrusion_content": {},
                "frame_mean_conf": 0.0,
                "objects_count": 0
                }

    detected_classes ={}
    alert_level = "GREEN" #defualt green, if noting dangerous detected

    confidences = []
    for detection in detections:
        class_name = str(detection.get("class", "unknown")).lower()

        confidence = float(detection.get("confidence", 0.0))
        confidences.append(confidence)

        if class_name in detected_classes:
            detected_classes[class_name] += 1

        else:
            detected_classes[class_name] = 1

        if class_name in critical_classes:
            alert_level = "CRITICAL"

        elif class_name in amber_classes and alert_level != "CRITICAL":
            alert_level = "AMBER"

    # check if any are in critical or amber
    has_critical_or_amber = any(
        class_name in critical_classes or class_name in amber_classes
        for class_name in detected_classes.keys()
    )

    frame_mean_conf = float(np.mean(confidences))
    # If NO critical or amber classes, return False for intrusion
    if not has_critical_or_amber:
        return {
            "intrusion": False,
            "alert_level": "GREEN",
            "intrusion_content": detected_classes,
            "frame_mean_conf":frame_mean_conf,
            "objects_count": int(sum(detected_classes.values()))
        }

    intrusion_response = {
        "intrusion": True,
        "alert_level": alert_level,
        "intrusion_content": detected_classes,
        # per class counts { "person": 1, "dog": 2, "car": 1 }
        "frame_mean_conf": frame_mean_conf,
        "objects_count": int(sum(detected_classes.values()))
        # total objects
    }

    return intrusion_response
