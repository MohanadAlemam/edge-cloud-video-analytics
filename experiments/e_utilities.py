import json
import pandas as pd

def build_dataframe(data_path):
    """
    Reads the raw JSON data from history metrics and builds a pandas dataframe.

    :param data_path: path to the history metrics data
    :return: structured dataframe
    """

    raw_data = json.load(open(data_path))

    rows = []
    for row in raw_data:
        edge = row.get("edge_metrics", {})
        network = row.get("network_metrics", {})
        intrusion = row.get("intrusion_metrics", {})
        cloud = row.get("cloud_metrics", {})

        rows.append({
            # frame info
            "frame_index": edge.get("frame_index"),
            "timestamp": edge.get("timestamp"),

            # edge
            "edge_inference_ms": edge.get("edge_inference_ms"),
            "avg_edge_inference_ms": edge.get("avg_edge_inference_time"),
            "total_frames_processed": edge.get("total_frames_processed"),
            "heuristic_frames_dropped": edge.get("heuristic_frames_dropped"),
            "heuristic_drop_ratio": edge.get("heuristic_drop_ratio"),
            "cloud_avoidance_ratio": edge.get("cloud_avoidance_ratio"),

            # network
            "frames_sent_to_cloud": network.get("total_frames_to_cloud"),
            "mb_sent_to_cloud": network.get("total_m_bytes_sent_to_cloud"),

            # intrusion detection
            "intrusion": intrusion.get("intrusion"),
            "alert_level": intrusion.get("alert_level"),
            "objects_count": intrusion.get("objects_count"),
            "frame_mean_conf": intrusion.get("frame_mean_conf"),

            # cloud
            "round_trip_time_ms": cloud.get("round_trip_time"),
            "avg_rt_ms": cloud.get("avg_rt_ms"),
            "cloud_infer_ms": cloud.get("cloud_infer_ms"),
            "avg_cloud_infer_ms": cloud.get("average_cloud_inference_ms"),
        })
    return pd.DataFrame(rows)