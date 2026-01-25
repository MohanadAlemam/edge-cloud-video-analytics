
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

    response.raise_for_status()
    # Error handling for HTTP requests. to avoid failed request or errors (eg 500, 404, 401) being treated as inference

    # parse response body as JSON and return the resulting Python object eg dictionary
    return response.json()


# 2. Orchestrator function to run the edge pipeline
def run_edge_pipeline(
        video_path:str,
        cloud_server_url:str,
        frame_skip: int = 20,
        resize_width:int = 640,
        jpeg_quality:int = 80,
        frame_change_threshold:float = 20.0,
        debug_mode:bool = False,
        debug_frames:int = 5,
):
    """"
    Orchestration function for edge pipeline to organise and manage edge processes.

    - Generate frames using video_frames_generator
    - Preprocess including resize and grayscale conversion
    - Decide whether to send. simple change in frame filter
    - Encode to JPEG bytes and send if interesting
    - Send the interesting frames to the could serve and receives responses, print results and simple local metrics.

    """
    # 1. GENERATING FRAMES FROM THE VIDEO
    print(f"Edge initiating processing pipeline. Server: {cloud_server_url}")

    frames_seen = 0
    previous_gray = None
    frames_sent = 0

    for frame_index, timestamp, frame in video_frames_generator(video_path, skip_frames=frame_skip):
        frames_seen += 1

        print (f"Edge frame {frame_index}, timestamp: {timestamp:.3f} second, (seen {frames_seen})")


        # 2. PREPROCESSING FRAMES
        smaller_frame = resize_frame(frame, target_width=resize_width)
        # resize the frame
        gray_and_small = covert_to_grayscale(smaller_frame)
        # greyscale the frame


        # 3. DECIDING IF THE FRAME IS INTERESTING BASED ON HEURISTIC CHANGE
        significant_change_detected, change_score = interesting_frames(previous_frame = previous_gray,
                                                                   current_frame = gray_and_small,
                                                                   difference_threshold = frame_change_threshold)

        print(f"Edge frame change score = {change_score:.2f}.\nIs the frame interesting? {significant_change_detected}")
        # Call interesting_frames function and assess the change in the frames


        # 5. SEND ONLY INTERESTING FRAMES TO THE CLOUD FOR INFERENCE
        if significant_change_detected:
            # in case the frame is interesting ie significant change occurs
            # we need to send this frame to the cloud for inference, colored version of the frame as it has more info

            jpeg_payload = encode_to_jpeg_bytes(frame=smaller_frame, quality=jpeg_quality)
            # prepare the colored version of the frame as it has more info

            send_start = time.time()
            # record the sending time and send the frame
            try:
                cloud_response = feed_cloud_jpeg(cloud_server_url = cloud_server_url,
                                                 jpeg_bytes = jpeg_payload
                                                 )
                round_trip_time = time.time() - send_start * 1000.0 # measure round trip in ms
                frames_sent += 1 # up the frames sent counter

                print(f"Edge sent frame{frame_index} .Round trip time: {round_trip_time:.2f} ms."
                      f"\nServer response: {cloud_response}")

            except Exception as e:
                print(f"Edge Error related to sending frame: {frame_index}, {e}")
                # handling errors

        previous_gray = gray_and_small
        # current frame becomes previous

        if debug_mode and frames_seen == debug_frames:
            break

    # give a summary at the end
    print(f"Edge processing completed. Frames seen: {frames_seen}, Frames sent: {frames_sent}")

























