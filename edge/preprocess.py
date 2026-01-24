# Resize encode filtering and maybe optional edge model
import cv2
import numpy as np
from typing import Tuple, Optional

def resize_frame(
        frame: np.ndarray,
        target_width: Optional[int] = None,
        # either an int or None
        target_height: Optional[int]=None
) -> np.ndarray: #the function returns a NumPy array

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


