#-----------------------------------------------------------------------------------------------------------------------
# COMMON PROCESSES
#-----------------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np

# Function to draw detections of frames
def annotate_frame(resized_frame: np.ndarray, cloud_response: dict):
    """
    Draw bounding boxes, and shows labels on each frame based on the cloud_response.

    :param resized_frame: frame to draw on, the small/resized frame
    :param cloud_response: dict returned from cloud inference which contains 'detections'
    :return: annotated frame which can be written to video or displayed
    """

    detections = cloud_response.get("detections", [])
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
            # (0, 255, 0) make the color in BGR format = green, and 2 is the thickness of the borders

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
