def frame_content_count(detections: list[dict]) -> dict:
    """
    Counts the number of vehicles and pedestrians in a frame.

    :param detections: detections object for a single frame
    :return: a dictionary with vehicle_count, pedestrian_count
    """
    vehicle_classes = {"car","motorcycle","bus","train","truck","boat", "bicycle"}
    pedestrian_class = {"person", "dog"}

    vehicle_count = 0
    pedestrian_count = 0

    for detection in detections:
        class_name = detection.get("class", "unknown").lower()
        if class_name in vehicle_classes:
            vehicle_count += 1
        elif class_name in pedestrian_class:
            pedestrian_count += 1

    return {"vehicle_count": vehicle_count, "pedestrian_count": pedestrian_count}
