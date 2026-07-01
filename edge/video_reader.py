import cv2

def video_frames_generator (video_path, skip_frames=0):
    """
    generator that yields frames from a video file.

    :param video_path: path to video file.
    :param skip_frames: number of frames to skip between yielded frames. with = 0 means produce all frames.

    :return:
       tuple: (frame_id, frame_timestamp, frame)
            frame_id (int): Index of the frame in the video.
            frame_timestamp (float): Timestamp of the frame in seconds.
            frame (ndarray): frame/image array in BGR color format.
    """
    capture = cv2.VideoCapture(video_path) # read the video on path
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    # if opening failed raise error

    frame_id = 0

    while True:
        return_value, frame = capture.read()
        # as long as return_value is true continue reading frames
        if not return_value:
            break

        if(frame_id % (skip_frames + 1)) == 0:
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC) # get current frame position in milliseconds
            frame_timestamp = timestamp_ms / 1000.0 # in seconds

            yield frame_id, frame_timestamp, frame
            # produces frame id, timestamp in seconds and the frame

        frame_id += 1
    capture.release()  # release the video processing including: file handles, decoders