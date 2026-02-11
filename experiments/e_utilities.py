import json
import pandas as pd
import matplotlib.pyplot as plt

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



def bandwidth_bar_plot(cloud_mb, edge_mb):
    """
    Plots the bandwidth bar graph.

    :param cloud_mb: cloud MB
    :param edge_mb: edge MB
    :return: bandwidth bar graph
    """
    cloud_mb = float(cloud_mb.iloc[0])
    edge_mb = float(edge_mb.iloc[0])

    # Simple bar plot
    plt.bar(["cloud_only", "edge_only"], [cloud_mb, edge_mb], color=["blue", "orange"])
    plt.ylabel("Total MB sent to cloud")
    plt.title("Bandwidth Usage Comparison (last snapshot)")
    plt.text(0, cloud_mb + 0.01 * cloud_mb, f"{cloud_mb:.2f}", ha="center")
    plt.text(1, edge_mb + 0.01 * cloud_mb, f"{edge_mb:.2f}", ha="center")
    plt.show()



def align_data(edge_df:pd.DataFrame, cloud_df:pd.DataFrame, on: str = "frame_index"):
    """
    Align edge and cloud perframe metrics into a single dataframe.

    :param edge_df: edge dataframe
    :param cloud_df: cloud dataframe
    :param on: Column to align on. Default "frame_index" can be "timestamp".
    :return:
    """
    edge_columns = [on, "objects_count","frame_mean_conf", "edge_inference_ms"]
    edge = edge_df[edge_columns].copy() # make a copy

    # rename the columns for better alignment
    edge = edge.rename(columns={
        "objects_count":"count_e",
        "frame_mean_conf":"avg_conf_e",
        "edge_inference_ms":"lat_e",})

    cloud_columns = [on,"objects_count", "frame_mean_conf", "round_trip_time_ms"]
    cloud = cloud_df[cloud_columns].copy()

    cloud = cloud.rename(columns={
        "objects_count":"count_c",
        "frame_mean_conf":"avg_conf_c",
        "round_trip_time_ms":"lat_c",
    })

    merged = pd.merge(edge, cloud, on=on, how="inner")
    # inner for 1 to 1 comparison (keeps only frames present in both)

    merged= merged.sort_values(by=on)
    merged = merged.reset_index(drop=True)

    return merged



def plot_time_series(merged_df:pd.DataFrame, on : str = "frame_index", smooth_window : int = 0):
    """
    plots time serise/ frame-wise series of the number of the per frame confidence and latency.

    :param merged_df: the merged dataframe
    :param on: the variable to merge the two dataframes
    :param smooth_window: To compute the rolling mean to smooth the line graph.
    :return: 
    """
    df = merged_df.copy()

    if smooth_window >1:
        # Apply simple rolling mean smoothing for readability -- min_periods=1 to avoid nan
        df["count_e_s"] = df["count_e"].rolling(smooth_window, min_periods=1, center=True).mean()
        df["count_c_s"] = df["count_c"].rolling(smooth_window, min_periods=1, center=True).mean()
        df["conf_e_s"] = df["avg_conf_e"].rolling(smooth_window, min_periods=1, center=True).mean()
        df["conf_c_s"] = df["avg_conf_c"].rolling(smooth_window, min_periods=1, center=True).mean()
        df["lat_e_s"] = df["lat_e"].rolling(smooth_window, min_periods=1, center=True).mean()
        df["lat_c_s"] = df["lat_c"].rolling(smooth_window, min_periods=1, center=True).mean()
    else:
        df["count_e_s"], df["count_c_s"] = df["count_e"], df["count_c"]
        df["conf_e_s"], df["conf_c_s"] = df["avg_conf_e"], df["avg_conf_c"]
        df["lat_e_s"], df["lat_c_s"] = df["lat_e"], df["lat_c"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

    # 1. object counts
    axes[0].plot(df[on], df["count_e_s"], label="edge counts", color="tab:orange")
    axes[0].plot(df[on], df["count_c_s"], label="cloud counts", color="tab:blue")
    axes[0].set_ylabel("Number of Objects", fontsize=15)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].set_title("Per-frame object counts: edge vs. cloud", fontsize=15, fontweight="bold")

    # 2.mean confidence
    axes[1].plot(df[on], df["conf_e_s"], label="edge avg conf", color="tab:orange")
    axes[1].plot(df[on], df["conf_c_s"], label="cloud avg conf", color="tab:blue")
    axes[1].set_ylabel("Avg confidence", fontsize=15)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].set_title("Per-frame avg confidence: edge vs. cloud", fontsize=15, fontweight="bold")

    # 3. latency -- edge inference vs cloud round trip
    axes[2].plot(df[on], df["lat_e_s"], label="edge latency (ms)", color="tab:orange")
    axes[2].plot(df[on], df["lat_c_s"], label="cloud latency (ms)", color="tab:blue")
    axes[2].set_ylabel("Latency (ms)", fontsize=15)
    axes[2].set_xlabel(on, fontsize=16)
    axes[2].legend(loc="upper right", fontsize=10)
    axes[2].set_title("Per-frame latency: edge vs. cloud", fontsize=15, fontweight="bold")

    plt.tight_layout()
    plt.show()