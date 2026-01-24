# a lightweight edge model to reduce traffic to the cloud

# import the heuristic from preprocess as a tigger to wake the lightweight model
from preprocess import interesting_frames

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import time



