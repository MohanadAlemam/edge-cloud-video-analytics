
import requests
# HTTP client, sends data to the cloud
import cv2
# for frames/image encoding
import argparse
# Commandline argument parsing
import os
# for environment variables
import time
# Timing utilities and latency measurement

from video_reader import video_frames_generator

from preprocess import (resize_frame,
                        covert_to_grayscale,
                        encode_to_jpeg_bytes,
                        interesting_frames)


# 1. a function to feed the could frames
def feed_cloud_jpeg(
        cloud_server_url:str,
        jpeg_bytes:bytes,
        requests_timeout_sec: int =10
):
    """"
    Send the in-memory jpeg bytes to cloud server for inference.

    it only does the HTTP request and return parsed JSON.
    """
    # prepare multipart or form data payload, field "image" with (filename, bytes, mime-type/ media type eg JPEG image)
    files_payload = {
        "image": ("frame.jpg", jpeg_bytes, "image/jpeg")
    }
    # will post the in-memory JPEG as a multi-part file/payload to server/infer
    response = requests.post(cloud_server_url.rstrip("/") + "/infer",
                             files=files_payload, timeout=requests_timeout_sec)
    # normalize URL by removing slashes if present and append the /infer endpoint then POST files

    



































#api_url = "http://localhost:5000/hello"
api_url = "http://10.0.0.3:5000/hello"
response = requests.get(api_url)
rd = response.json()
pretty = json.dumps(rd, indent=4)
print(pretty)
