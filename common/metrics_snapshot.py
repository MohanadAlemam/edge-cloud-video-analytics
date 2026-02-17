from pathlib import Path
import json


def write_metrics_snapshot(edge_metrics,
                           network_metrics,
                           intrusion_metrics,
                           cloud_metrics,
                           m_output_dir="output"):
    """
    This function writes the metrics snapshot per frame to a file

    :param edge_metrics: edge metrics collected for the current frame
    :param cloud_metrics: cloud metrics collected for the current frame
    :param network_metrics: network metrics collected for the current frame
    :param intrusion_metrics: intrusion metrics collected for the current frame.
    :return: None. write to hard disk
    """
    METRICS_PATH = Path(m_output_dir) / "metrics_history.json"
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    snapshot_metrics = {
        "edge_metrics": edge_metrics,
        "network_metrics": network_metrics,
        "intrusion_metrics": intrusion_metrics,
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
        MAX_HISTORY_LENGTH = 10000
        # to avoid getting a massive file
        if len(data) > MAX_HISTORY_LENGTH:
            data = data[-MAX_HISTORY_LENGTH:] # keep the last 500

        # auto write
        temporary = METRICS_PATH.with_suffix(".tmp")
        # creat a temp file
        temporary.write_text(json.dumps(data, indent=2))
        # write to a temporary file first 'metrics_history.tmp' before replacing the real JSON file
        temporary.replace(METRICS_PATH)
        print(f"Metrics snapshot written to: {METRICS_PATH}\n")

    except Exception as e:
        print(f"Error: Failed to write the metrics snapshot  {e}")