import requests
# HTTP client, sends data to the cloud
import argparse
# Commandline argument parsing
import time
import numpy as np


from video_reader import video_frames_generator

from preprocess import (resize_frame,
                        convert_to_grayscale,
                        encode_to_jpeg_bytes,
                        is_frame_interesting)

from edge_model import EdgeModel #import the edge model to conduct lightweight on-edge processing


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
    Orchestration function for edge pipeline to organize and manage edge processes.

    - Generate frames using video_frames_generator
    - Preprocess including resize and grayscale conversion
    - Decide whether to send. simple change in frame filter
    - Encode to JPEG bytes and send if interesting
    - Send the interesting frames to the could serve and receives responses, print results and simple local metrics.

    """
    # 1. GENERATING FRAMES FROM THE VIDEO
    print(f"Edge initiating processing pipeline. Server: {cloud_server_url}")

    total_frames_processed = 0
    previous_grey = None

    frame_sent_to_cloud = 0
    cloud_round_trips_ms = []  # track all cloud round trips time ms
    edge_model_inferences_ms = [] # collects all edge model inference times
    total_bytes_sent_to_cloud = 0

    # 1.1 Create an EdgeModel instance once before the loop.
    edge_model = EdgeModel()

    for frame_index, timestamp, colored_frame in video_frames_generator(video_path, skip_frames=frame_skip):
        total_frames_processed += 1

        print (f"Edge frame {frame_index}, timestamp: {timestamp:.3f} second, (seen {total_frames_processed})")


        # 2. PREPROCESSING FRAMES
        # resize the frame and  greyscale the frame
        smaller_frame = resize_frame(colored_frame, target_width=resize_width)
        grey_and_small = convert_to_grayscale(smaller_frame)

        # Skip change detection for the first one
        if previous_grey is None:
            previous_grey = grey_and_small
            continue # stop and go to the next iteration

        # 3. DECIDING IF THE FRAME IS INTERESTING BASED ON HEURISTIC CHANGE
        significant_change_detected, change_score = is_frame_interesting(previous_frame = previous_grey,
                                                                   current_frame = grey_and_small,
                                                                   difference_threshold = frame_change_threshold)

        print(f"Edge frame change score = {change_score:.2f}.\nIs the frame interesting? {significant_change_detected}")
        # Call interesting_frames function and assess the change in the frames

        # Dont send to the cloud if nothing changed
        if not significant_change_detected:
            print(f"As per the pixels values no significant change in the video feed was detected."
                  f"\nNo further processing will be performed.")
            previous_grey = grey_and_small
            continue # stop and go to the next iteration

        # 4. LIGHTWEIGHT EDGE MODEL

        # if the fames are significantly different try the edge model (yolo) before sending to the cloud
        else:
            send_to_cloud, edge_inference_time_ms = edge_model.yolo_decision(colored_frame)

            edge_model_inferences_ms.append(edge_inference_time_ms) # register this edge inference time

            # 5. SEND ONLY INTERESTING FRAMES TO THE CLOUD FOR INFERENCE
            if send_to_cloud:

                # send the colored frame to the cloud for inference, as it has more info

                jpeg_payload = encode_to_jpeg_bytes(frame=smaller_frame, quality=jpeg_quality)
                # prepare the colored version of the frame as it has more info

                total_bytes_sent_to_cloud += len(jpeg_payload) # accumulate the total bytes

                send_start = time.time()
            # record the sending time and send the frame
                try:
                    cloud_response = feed_cloud_jpeg(cloud_server_url = cloud_server_url,
                                                 jpeg_bytes = jpeg_payload
                                                 )
                    round_trip_time = (time.time() - send_start) * 1000.0 # measure round trip in ms
                    frame_sent_to_cloud += 1 # up the frames sent counter
                    cloud_round_trips_ms.append(round_trip_time) # collect ms round trip time

                    print(f"Edge sent frame{frame_index} .Round trip time: {round_trip_time:.2f} ms."
                          f"\nServer response: {cloud_response}")

                except Exception as e:
                    print(f"Edge Error related to sending frame: {frame_index}, {e}")
                    # handling errors

        previous_grey = grey_and_small
        # current frame becomes previous

        if debug_mode and total_frames_processed == debug_frames:
            break

    # 6. Metrics
    frame_drop_ratio = 1 - (frame_sent_to_cloud / total_frames_processed) if total_frames_processed > 0 else 0
    # a metric to show the usefulness of the edge intelligence/processing and dropping of frames
    average_round_trip = np.mean(cloud_round_trips_ms) if cloud_round_trips_ms else 0
    slowest_round_trip = max(cloud_round_trips_ms) if cloud_round_trips_ms else 0

    average_edge_inference_ms = np.mean(edge_model_inferences_ms) if edge_model_inferences_ms else 0
    total_bytes_sent_kb = total_bytes_sent_to_cloud / 1024 if total_bytes_sent_to_cloud else 0 # convert to kb

    edge_metrics = {
        "total_frames_processed": total_frames_processed,
        "frame_drop_ratio": frame_drop_ratio,
        "avg_edge_inference_time": average_edge_inference_ms,
    }


    network_bandwidth_metrics ={
        "total_frames_to_cloud": frame_sent_to_cloud,
        "total_bytes_sent_to_cloud": total_bytes_sent_kb,
    }

    cloud_metrics = {
        "avg_rt_ms": average_round_trip,
        "slowest_rt_ms": slowest_round_trip,
        "cloud_requests_handled": XXXXXX,
        "avg_cloud_inference_time": XXX,
    }

    # give a summary at the end
    print(f"Edge processing completed:"
          f"\n\nFrames seen: {total_frames_processed}"
          f"\n\nFrames sent to the cloud: {frame_sent_to_cloud}"
          f"\n\nDrop ratio: {frame_drop_ratio}")

    return edge_metrics, network_bandwidth_metrics, cloud_metrics


# 7. Command-Line Interface (CLI) to run the program from the terminal and control it using text commands and flags

if __name__ == "__main__":
    # Runs only when execute python edge/client.py in the terminal.
    # Create a CLI parser so users can run this script from the terminal
    parser = argparse.ArgumentParser(
        description="Edge pipeline: generating, preprocessing and sending interesting frames to the cloud for inference."
    )
    # Required path to the input video file
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the video file.")

    # Optional tuning parameters exposed as CLI flags
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000",
                        help="Cloud server URL")

    parser.add_argument("--skip_interval", type=int, default=0,
                        help="Frame Skip: consider and send every (skip+1) frame, interval between frames")
    parser.add_argument("--resize_width", type=int, default=640,
                        help="Resized frame width (px)")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG quality (px) 0-100")
    parser.add_argument("--change_threshold", type=float, default=15.0,
                        help="Change Threshold: the mean pixel difference in the frame to consider the frame interesting.")
    parser.add_argument("--debug_mode", type=bool, default=False,
                        help = "Debug mode: to conduct a a test by processing and sending only a a specific number of frames.")
    parser.add_argument("--debug_frames", type=int, default=5,
                        help="Debug frames: number of frame for the debug mode.")

    # parse the CLI arguments into the pipline argument object
    arguments = parser.parse_args()

    # call the main orchestrator function
    run_edge_pipeline(
        video_path = arguments.video_path,
        cloud_server_url = arguments.server_url,
        frame_skip = arguments.skip_interval,
        resize_width = arguments.resize_width,
        jpeg_quality = arguments.quality,
        frame_change_threshold = arguments.change_threshold,
        debug_mode = arguments.debug_mode,
        debug_frames = arguments.debug_frames
    )






















