from pathlib import Path
import json


def write_metrics_snapshot(edge_metrics, network_metrics, content_metrics, cloud_metrics= {}):
    """
    This function writes the metrics snapshot per frame to a file

    :param edge_metrics: edge metrics collected for the current frame
    :param cloud_metrics: cloud metrics collected for the current frame
    :param network_metrics: network metrics collected for the current frame
    :param content_metrics: content metrics collected for the current frame.
    :return: None. write to hard disk
    """
    METRICS_PATH = Path("output/metrics_history.json")

    if not cloud_metrics:
        snapshot_metrics = {
            "edge_metrics": edge_metrics,
            "network_metrics": network_metrics,
            "content_metrics": content_metrics,
        }
    elif isinstance(cloud_metrics, dict):
        snapshot_metrics = {
            "edge_metrics": edge_metrics,
            "network_metrics": network_metrics,
            "content_metrics": content_metrics,
            "cloud_metrics": cloud_metrics
        }
    try:
        if METRICS_PATH.exists():
            raw = METRICS_PATH.read_text().strip()
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = [data]
            else:
                data = []
        else:
            data = []

        # append the snapshot history and remove old logs
        data.append(snapshot_metrics)
        MAX_HISTORY_LENGTH = 500
        if len(data) > MAX_HISTORY_LENGTH:
            data = data[-MAX_HISTORY_LENGTH:] # keep the last 500

        # auto write
        temporary = METRICS_PATH.with_suffix(".tmp")
        # creat a temp file
        temporary.write_text(json.dumps(data, indent=2))
        # write to a temporary file first 'metrics_history.tmp' before replacing the real JSON file
        temporary.replace(METRICS_PATH)
        print(f"\nMetrics snapshot written to: {METRICS_PATH}")

    except Exception as e:
        print(f"Error: Failed to write the metrics snapshot  {e}")