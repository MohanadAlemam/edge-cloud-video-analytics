# Resize encode filtering and maybe optional edge model
import cv2
import numpy as np
from typing import Tuple, Optional


# 1. Resizing
def resize_frame(
        frame: np.ndarray,
        target_width: Optional[int] = None,
        # either an int or None
        target_height: Optional[int]=None
) -> np.ndarray: #the function returns a NumPy array
    """
    Resize the frame to fit the target size, or return the original frame if no dimensions are provided.

    frame: np.ndarray of the frame to be resized.
    target_width: Optional[int]
    target_height: Optional[int]
    """

    height, width = frame.shape[:2]
    if target_width is None and target_height is None:
        return frame

    if target_width is None:
        # calculate width from height
        scale = target_height / float(height)
        target_width = int(width * scale)

    elif target_height is None:
        # calculate target height from width
        scale = target_width / float(width)
        target_height = int(height * scale)

    # calling OpenCV resize to re-size the frame based on the new width and height
    resized_frame = cv2.resize(frame,
                               (int(target_width), int(target_height)),
                               interpolation=cv2.INTER_AREA
                               # interpolation for how to resize the frame ,use best choice
                               )
    return resized_frame


# 2. convert to grayscale (useful for processing in the edge)
def covert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convert the frame to grayscale and return it.

    :param frame: frame to be converted to grayscale.
    :return: grayscale frame.
    """
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayscale_frame


# 3. Encode frames to JPEG
def encode_to_jpeg_bytes(frame: np.ndarray, quality:int = 80) -> bytes:
    """
    Encode the frame (BGR or grayscale) to jpg bytes in memory, and return it.
    :param frame: frame to be encoded.
    :param quality: quality of the encoded frame.
    :return: encoded frame.
    """

    encode_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    # Quality controls compression and returns row bytes

    done, encoded_frame = cv2.imencode(".jpg", frame, encode_parameters)
    # monitor failure in encoding
    if not done:
        raise RuntimeError("Failed to encode frame as JPEG.")

    return encoded_frame.tobytes()


# 3. Filter and select interesting frames, frame difference including environmental changes and motion detection
def interesting_frames(
        previous_frame: Optional[np.ndarray], # previous grayscale frame
        current_frame: np.ndarray, # current gray frame
        difference_threshold: float = 15.0
) -> Tuple[bool, float]:
    """
    Basic filtering function for interesting frames using basic difference algorithm.

    :param previous_frame: previous frame in grayscale
    :param current_frame: current frame in grayscale.
    :param difference_threshold: difference threshold in pixels.
    :return: (is_interesting (True or False), motion_score)
    """
    if previous_frame is None:
        return True, float("Inf")
    # no previous frame consider current interesting, always sent the first frame

    absolute_difference = cv2.absdiff(previous_frame, current_frame)
    # calculate the difference / environmental changes and motion detection
    motion_score = float(np.mean(absolute_difference))

    significant_motion_detected = motion_score >= difference_threshold
    # Compare to the tolerance threshold

    return significant_motion_detected, motion_score

