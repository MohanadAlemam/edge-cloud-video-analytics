# STEP 5. ADVANCED COMMAND LINE CONFIGURATION (CLI FLAGS)

## ORCHESTRATOR FLAGS

| Flag | Type | Default                       | Meaning                             |
|------|------|-------------------------------|-------------------------------------|
| `--video_path` | str | required                      | Path to input video.                |
| `--server_url` | str | example: http://10.0.0.3:5000 | Currently running Cloud server URL. |
| `--skip_interval` | int | 5                             | Number of frames skipped.           |
| `--resize_width` | int | 640                           | Frame resize width.                 |
| `--quality` | int | 80                            | JPEG quality.                       |
| `--heuristic_threshold` | float | 5.0                           | Motion detection threshold.         |
| `--edge_conf_threshold` | float | 0.70                          | Edge confidence threshold.          |
| `--m_output_dir` | str | output                        | Output directory.                   |
| `--debug_mode` | flag | False                         | Debug mode.                         |
| `--store_video` | flag | False                         | Save annotated video.               |

## CLOUD SERVER FLAGS

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--model_path` | str | `./models/yolov8s.pt` | Cloud model path |
| `--port` | int | 5000 | Flask server port |
