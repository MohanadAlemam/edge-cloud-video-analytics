import cv2
import requests
# HTTP client, sends data to the cloud
import argparse
# Commandline argument parsing
import time
from datetime import datetime
import numpy as np
import json
from pathlib import Path
import threading

# IMPORT HELPER FUNCTIONS
from .video_reader import video_frames_generator
from .preprocess import (resize_frame,
                        convert_to_grayscale,
                        encode_to_jpeg_bytes,
                        is_frame_interesting)

from .edge_model import EdgeModel
#import the edge model to conduct lightweight on edge processing
from common.visualize import annotate_frame
# load the annotation function
from .cloud_feeder import feed_cloud_jpeg
# load the feeder function
from common.metrics_snapshot import write_metrics_snapshot
# import the helper to update the metrics for the dashboard

VIDEO_OUTPUT_PATH = "output/annotated_output.mp4"
# the name of the output file if no realtime/ live display

#-----------------------------------------------------------------------------------------------------------------------
# ORCHESTRATOR FUNCTION TO RUN AND MANGE THE PIPELINE
#-----------------------------------------------------------------------------------------------------------------------

def orchestrator_run_pipeline(
        video_path:str,
        cloud_server_url:str,
        frame_skip: int = 5,
        resize_width:int = 640,
        jpeg_quality:int = 80,
        heuristic_threshold: float = 5.0,
        edge_conf_threshold: float = 0.70,
        live_display:bool = False,
        m_output_dir = "output", # where to write metrics snapshots
        debug_mode:bool = False,
        debug_frames:int = 5,
):
    """
    Orchestration function for edge pipeline to organize and manage processes.

    - Generate frames using video_frames_generator.
    - Preprocess including resize and grayscale conversion.
    - Decide whether to send. simple change in frame filter.
    - Encode to JPEG bytes and send if interesting
    - Send the interesting frames to the could serve and receives responses, print results and simple local metrics.
    """
    # 1. GENERATING FRAMES FROM THE VIDEO
    print(f"Edge initiating processing pipeline. Server: {cloud_server_url}")

    total_frames_processed = 0
    frames_processed_on_edge = 0
    frames_sent_to_cloud = 0

    previous_grey = None
    cloud_round_trips_ms = []
    # track all cloud round trips time ms
    edge_model_inferences_ms = []
    # collects all edge model inference times
    cloud_model_inferences_ms = []

    total_bytes_sent_to_cloud = 0
    round_trip_time = 0.0
    cloud_infer_ms = 0.0
    heuristic_drop_ratio = 0.0
    cloud_avoidance_ratio = 0.0

    # 1. Create an EdgeModel instance once before the loop.
    edge_model = EdgeModel(edge_conf_threshold = edge_conf_threshold)
    # Conduct asynchronous ultralytics loading and warm up
    threading.Thread(
        target=edge_model._load_edge_model,
        kwargs= {"background_warmup": True},
        daemon=True
    ).start()

    video_writer = None
    Path(VIDEO_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    # ensure parent dir exists: creates "output" if missing in case on not live processing

    # Collect the native video info
    capture = cv2.VideoCapture(video_path)
    original_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    capture.release()
    # get the native fps of the given video
    if original_fps <= 0.0:
        original_fps = 30.0
        # fallback to 30 fps

    effective_fps = original_fps / (frame_skip + 1)
    if effective_fps <= 0.0:
        effective_fps = 1.0
        # fallback to 1 fps

    # wait for the next frame ms
    wait_ms = max(1, int(1000 / effective_fps))
    # ms to pass to wait-key

    # Initialize safe defaults
    edge_metrics = {}
    network_metrics = {}
    cloud_metrics = {}
    intrusion_metrics = {}

    # 1. VIDEO DECODER
    try:
        for frame_index, timestamp, colored_frame in video_frames_generator(video_path, skip_frames=frame_skip):

            timestamp = float(timestamp)# time stamp is seconds
            total_frames_processed += 1
            print (f"Frame: {frame_index}. Timestamp: {timestamp:.2f} second.")

    # 2. PREPROCESSOR
            # resize the frame and  greyscale the frame
            smaller_frame = resize_frame(colored_frame, target_width=resize_width)
            grey_and_small = convert_to_grayscale(smaller_frame)

            # Seed first frame
            if previous_grey is None:
                previous_grey = grey_and_small
                continue # stop and go to the next iteration

    # 3. HEURISTIC FILTER - DECIDING IF THE FRAME IS INTERESTING BASED ON CHANGE
            significant_change_detected, change_score = is_frame_interesting(previous_frame = previous_grey,
                                                                             current_frame = grey_and_small,
                                                                             difference_threshold = heuristic_threshold)
            # Call interesting_frames function and assess the change in the frames
            if not significant_change_detected:
                print(f"Heuristic change assessment: Is frame interesting? {significant_change_detected}. "
                      f"Score {change_score:.2f}"
                      f"\nNo significant change in the video feed. No further processing will be performed.\n")
                previous_grey = grey_and_small
                continue # drop frame, and go to the next iteration

    # 4. LIGHTWEIGHT EDGE MODEL
            # if the fame is significantly different try the edge model (yolo) before sending to the cloud
            else:
                send_to_cloud, confidences_list, edge_response, edge_inference_ms = (
                    edge_model.edg_model_decision(smaller_frame))

                edge_model_inferences_ms.append(edge_inference_ms) # register this edge inference time
                intrusion_metrics = edge_response.get("intrusion_metrics",
                                                               {
                                                                   "intrusion": False,
                                                                   "alert_level": "GREEN",
                                                                   "intrusion_content": {},
                                                                   "frame_mean_conf": 0.0,
                                                                   "objects_count": 0
                                                               })
                # extract the count of vehicles and pedestrians, or 0  count
                if not send_to_cloud:
                    frames_processed_on_edge +=1
                # Annotate the frame
                display_frame = annotate_frame(smaller_frame, model_response=edge_response)
                print (f"Edge model: Send to cloud server? {send_to_cloud}."
                       f"\nInference time: {edge_inference_ms:.2f} ms.")

    # 5. OFFLOADING MODULE - SEND ONLY INTERESTING FRAMES TO THE CLOUD FOR INFERENCE
                if send_to_cloud:
                    # send the colored frame to the cloud for inference, as it has more info
                    jpeg_payload = encode_to_jpeg_bytes(frame=smaller_frame, quality=jpeg_quality)
                    total_bytes_sent_to_cloud += len(jpeg_payload) # accumulate the total bytes

                    send_start = time.time() #record the sending time and send the frame
    # 6. HEAVYWEIGHT CLOUD MODEL - GET THE CLOUD RESPONSE
                    try:
                        cloud_response = feed_cloud_jpeg(cloud_server_url = cloud_server_url,
                                                         jpeg_bytes = jpeg_payload)

                        round_trip_time = (time.time() - send_start) * 1000.0 # measure round trip in ms
                        frames_sent_to_cloud += 1 # up the frames sent counter
                        cloud_round_trips_ms.append(round_trip_time) # collect ms round trip time

                        intrusion_metrics = cloud_response.get("intrusion_metrics",
                                                               {
                                                                   "intrusion": False,
                                                                   "alert_level": "GREEN",
                                                                   "intrusion_content": {},
                                                                   "frame_mean_conf": 0.0,
                                                                   "objects_count": 0
                                                               })
                        ## to add a fallback for content count

                        # collect inference time ms
                        cloud_infer_ms = cloud_response.get("processing_time_ms",0.0)
                        cloud_model_inferences_ms.append(cloud_infer_ms)

                        display_frame = annotate_frame(smaller_frame, model_response=cloud_response)
                        # call annotate_frame function to place the detections on each frame

                        print(f"Cloud server: frame {frame_index} processed by the cloud model."
                              f"\nRound trip time: {round_trip_time:.2f} ms.")
                    except Exception as e:
                        print(f"Edge Error related to sending frame: {frame_index}, {e}")
                        # handling errors

    # 7. VIDEO DECODER - DRAW DETECTIONS/RESPONSES ON FRAMES
                        # No realtime display : write the annotated frames to an mp4 video, in output/
                if not live_display:
                    if video_writer is None:
                        # create the video writer for the once for the first frame
                        height, width = display_frame.shape[:2] #  # shape = height, width
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, effective_fps, (width, height))
                        print("Writing the annotated video to directory 'output'. Video writer opened:",
                              video_writer.isOpened())

                    video_writer.write(display_frame) # write the frame to the video

                if live_display:
                # show to user, OpenCV window
                    cv2.imshow("Frame", display_frame)
                    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                        break
                    # waitKey(1) waits 1 millisecond for a key press
                    # if user press q exit the loop/ stop live processing

            previous_grey = grey_and_small
            # current frame becomes previous

    # 8. METRICS AGGREGATOR

            frames_dropped = (total_frames_processed - frames_processed_on_edge - frames_sent_to_cloud) \
                if total_frames_processed > 0 else 0
            average_round_trip = np.mean(cloud_round_trips_ms) if cloud_round_trips_ms else 0
            average_edge_inference_ms = np.mean(edge_model_inferences_ms) if edge_model_inferences_ms else 0
            average_cloud_inference_ms = np.mean(cloud_model_inferences_ms) if cloud_model_inferences_ms else 0

            # Measures: fraction of  cloud traffic you avoided
            cloud_avoidance_ratio = 1 - (
                    frames_sent_to_cloud / total_frames_processed) if total_frames_processed > 0 else 0

            # fraction of filter drops
            heuristic_drop_ratio = frames_dropped / total_frames_processed if frames_dropped > 0 else 0

            slowest_round_trip = max(cloud_round_trips_ms) if cloud_round_trips_ms else 0
            total_bytes_sent_mb = total_bytes_sent_to_cloud / (1024 * 1024) if total_bytes_sent_to_cloud else 0
            # convert to kb

            # Edge metrics update per frame
            edge_metrics = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "edge_inference_ms": edge_inference_ms,
                "total_frames_processed": total_frames_processed,
                "heuristic_frames_dropped": frames_dropped,
                "cloud_avoidance_ratio": cloud_avoidance_ratio,
                "heuristic_drop_ratio": heuristic_drop_ratio,
                "avg_edge_inference_time": average_edge_inference_ms,
            }

            # Network metrics update per frame
            network_metrics = {
                "total_frames_to_cloud": frames_sent_to_cloud,
                "total_m_bytes_sent_to_cloud": total_bytes_sent_mb,
            }
            # Cloud Metrics
            cloud_metrics ={
                "round_trip_time": round_trip_time,
                "avg_rt_ms": average_round_trip,
                "cloud_infer_ms": cloud_infer_ms,
                "average_cloud_inference_ms": average_cloud_inference_ms,
                "slowest_rt_ms": slowest_round_trip,
            }

            # Write a snapshot of the metrics
            write_metrics_snapshot(
                edge_metrics = edge_metrics,
                cloud_metrics = cloud_metrics,
                network_metrics = network_metrics,
                intrusion_metrics = intrusion_metrics,
                m_output_dir = m_output_dir)

            if debug_mode and total_frames_processed == debug_frames:
                break

    finally:
        if video_writer is not None:
            video_writer.release()
            # Finalizes and closes the video file and  writes video metadata to disk

        if live_display:
            cv2.destroyAllWindows()
            # Close all OpenCV windows created by cv2.imshow()

    # give a summary at the end
    print(f"\nVideo analytics completed:"
          f"\n\nTotal frames seen: {total_frames_processed}"
          f"\n\nFrames sent to the cloud: {frames_sent_to_cloud}"
          f"\n\nHeuristic_drop_ratio: {heuristic_drop_ratio:.2f}"
          f"\n\nCloud avoidance ratio: {cloud_avoidance_ratio:.2f}\n")


    return edge_metrics, network_metrics, cloud_metrics, intrusion_metrics

#-----------------------------------------------------------------------------------------------------------------------
# Command-Line Interface (CLI) to run the program from the terminal and control it using text commands and flags
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Runs only when execute python edge/client.py in the terminal.
    # CLI parser
    parser = argparse.ArgumentParser(
        description="Run edge pipeline"
    )
    # Required path to the input video file
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the video file.")

    # Optional tuning parameters exposed as CLI flags
    parser.add_argument("--server_url", type=str, default="http://10.0.0.3:5000",
                        help="Cloud server URL")

    parser.add_argument("--skip_interval",
                        type=int, default=5,
                        help="Frame Skip: consider and send every (skip+1) frame, interval between frames. Default is 5.")
    parser.add_argument("--resize_width",
                        type=int, default=640,
                        help="Resized frame width (px)")
    parser.add_argument("--quality",
                        type=int, default=80,
                        help="JPEG quality (px) 0-100")
    parser.add_argument("--heuristic_threshold",
                        type=float, default=5.0,
                        help="Change Threshold: the mean pixel difference in the frame to consider the frame interesting.")
    parser.add_argument("--debug_frames",
                        type=int, default=5,
                        help="Debug frames: number of frame for the debug mode.")
    parser.add_argument("--edge_conf_threshold",
                        type=float, default=0.70,
                        help="Confidence threshold for edge model, decides whether to send the frame to the cloud or not."
                             "Default: 70.0)")

    parser.add_argument("--m_output_dir",
                        type=str, default= "output",
                        help="Path to the output directory for metrics snapshots.")

    # Modes : debug and realtime display
    parser.add_argument("--debug_mode",
                        action="store_true",
                        help = "Enable debug mode: to conduct a test by processing a specific number of frames.")
    parser.add_argument("--live_display",
                        action="store_true",
                        help="Enable real-time live display of predictions, default: save video."
                             "\n\nif 'False' saves frames to a video file.")

    # parse the CLI arguments into the pipline argument object
    arguments = parser.parse_args()


    # call the main orchestrator function
    orchestrator_run_pipeline(
        video_path = arguments.video_path,
        cloud_server_url = arguments.server_url,
        frame_skip = arguments.skip_interval,
        resize_width = arguments.resize_width,
        jpeg_quality = arguments.quality,
        heuristic_threshold= arguments.heuristic_threshold,
        edge_conf_threshold = arguments.edge_conf_threshold,
        m_output_dir = arguments.m_output_dir,
        live_display = arguments.live_display,
        debug_mode = arguments.debug_mode,
        debug_frames = arguments.debug_frames,
    )
#-----------------------------------------------------------------------------------------------------------------------















