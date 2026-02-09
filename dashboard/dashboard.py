#-----------------------------------------------------------------------------------------------------------------------
# EDGE CLOUD VIDEO ANALYTICS METRICS DASHBOARD
#-----------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import json

import time
from pathlib import Path


METRICS_PATH = Path("output//metrics_history.json")
# path to metrics file produced by orchestrator

st.set_page_config(page_title="Edge-Cloud Video Analytics Dashboard", layout="wide")
st.title("Metrics Dashboard: Edge-Cloud Video Analytics")
# main title on the dashboard

st.sidebar.header("Settings")
refresh_interval = st.sidebar.number_input(
    "Auto-refresh interval (seconds, 0 = off)",
    min_value=0,
    value=1,
    step=1
)

show_raw = st.sidebar.checkbox("Show raw JSON", value=False)
# checkbox to toggle showing raw JSON
max_runs = st.sidebar.number_input(
    "Max runs shown",
    min_value=1,
    value=5000,
    step=1)
# how many recent runs to show value=5000 = default

# Helper functions
def _load_history(path:Path, maximum_runs:int):
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
        # parse the file contents into Python objects
    except Exception as e:  # catch parsing / I/O errors
        st.error(f"Error loading the metrics JSON {e}")
        return []
    # convert it to a list if not a list
    if isinstance(data, list):
        return data[-maximum_runs:] # # returns the full list
    else:
        return [data]


def _build_dataframe(history:list[dict]):
    """
    build a pandas dataframe from history data which contain metrics.

    :param history: the raw history data
    :return: a pandas dataframe
    """
    df_rows = []
    for i, record in enumerate(history):
        edge = record.get("edge_metrics", {})
        network = record.get("network_metrics", {})
        cloud = record.get("cloud_metrics", {})
        intrusion = record.get("intrusion_metrics", {})

        df_rows.append({
            "run_index": i,
            "timestamp(s)": round(float(edge.get("timestamp", 0)), 2),
            "total_frames_seen": int(edge.get("total_frames_processed", 0)),
            "edge_inference_ms": round(float(edge.get("edge_inference_ms", 0)), 2),
            "edge_avg_infer_time(ms)": round(float(edge.get("avg_edge_inference_time", 0)), 2),
            "cloud_avoidance_ratio": round(float(edge.get("cloud_avoidance_ratio", 0)), 2),
            "heuristic_drop_ratio": round(float(edge.get("heuristic_drop_ratio", 0)), 2),
            "heuristic_frames_dropped": round(float(edge.get("heuristic_frames_dropped", 0)), 2),

            "cloud_infer_ms": round(float(cloud.get("cloud_infer_ms", 0)), 2),
            "average_cloud_inference_ms": round(float(cloud.get("average_cloud_inference_ms", 0)), 2),
            "round_trip_ms": round(float(cloud.get("round_trip_time", 0)), 2),
            "avg_rt_time(ms)": round(float(cloud.get("avg_rt_ms", 0)), 2),
            "slowest_rt(ms)": round(float(cloud.get("slowest_rt_ms", 0)), 2),
            "frames_sent": int(network.get("total_frames_to_cloud", 0)),
            "total_mb_sent": round(float(network.get("total_m_bytes_sent_to_cloud", 0)), 2),

            # content metrics
            "intrusion": intrusion.get("intrusion", False),
            "alert_level": intrusion.get("alert_level", "GREEN"),
            "intrusion_content": intrusion.get("intrusion_content", {}),
            "frame_mean_conf": round(float(intrusion.get("frame_mean_conf", 0.0)), 2),
            "objects_count": int(intrusion.get("objects_count", 0)),
        })

    if not df_rows:
        return pd.DataFrame()

    else:
        return pd.DataFrame(df_rows)


def main():
    """
    live metrics dashboard for edge, cloud, network and intrusion status metrics.

    """
    history = _load_history(METRICS_PATH, max_runs)
    if not history:
        st.warning("No history data. Please run the orchestration to produce output and metrics.")
        # run the waring a return/ do nothing
        return

    dataframe = _build_dataframe(history)

    if dataframe.empty:
        st.warning("The metics file is empty. Please run the orchestration to produce metrics.")
        return

    # pick the most recent row for KPI display
    latest_run = dataframe.iloc[-1]

    st.subheader("Edge Metrics")
    k0, k1, k2, k3, k4 = st.columns(5)
    k0.metric("Timestamp (s)", latest_run.get("timestamp(s)"))
    k1.metric("Frames Seen", latest_run["total_frames_seen"])
    k2.metric("Heuristically Dropped", latest_run["heuristic_frames_dropped"])
    k3.metric("Heuristic Drop Ratio", latest_run["heuristic_drop_ratio"])
    k4.metric("Avg Infer Time (ms)", latest_run["edge_avg_infer_time(ms)"])

    # Content metrics
    st.markdown("----")
    st.subheader("Intrusion Status")
    k5, k6, k7, k8, k9 = st.columns(5)
    k5.metric("Intrusion Detected", latest_run["intrusion"])
    k6.metric("Alert Level", latest_run["alert_level"])
    k7.metric("Avg Frame Confidence", latest_run["frame_mean_conf"])
    k8.metric("Total Objects", latest_run["objects_count"])


    content = latest_run.get("intrusion_content", {})
    if content:
        content_list =list(content.items())
        content_df = pd.DataFrame(content_list, columns=["Top Intruders", "Count"])
        #convert to pd df
        content_df = content_df.sort_values("Count", ascending=False)
        # sort the intrusion content
        #k7.subheader("Threat categories")
        k9.table(content_df.head(3).set_index("Top Intruders"))
    # show top 3
    else:
        k9.info("No source of threat.")

    # Cloud metrics
    st.markdown("----")
    st.subheader("Cloud and Network Metrics")
    k10, k11, k12, k13, k14 = st.columns(5)
    k10.metric("Frames -> Cloud", latest_run["frames_sent"])
    k11.metric("Bandwidth Usage (MB)", latest_run["total_mb_sent"])
    k12.metric("Cloud Avoidance Ratio", latest_run["cloud_avoidance_ratio"])
    k13.metric("Avg Round-Trip Latency (ms)", latest_run["avg_rt_time(ms)"])
    k14.metric("Avg Infer Time (ms)", latest_run["average_cloud_inference_ms"])

    # plot and charts
    st.markdown("----")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Edge Heuristic Drop Ratio")
        st.area_chart(dataframe.set_index("run_index")["heuristic_drop_ratio"],
                      x_label="Run",
                      y_label="Drop Ratio")
        # simple line chart for drop ratio
    with c2:
        st.subheader("Edge Infer Time")
        st.line_chart(dataframe.set_index("run_index")["edge_inference_ms"],
                      x_label="Run",
                      y_label="Infer Time (ms)")
    with c3:
        st.subheader("Network Usage")
        st.bar_chart(dataframe.set_index("run_index")["total_mb_sent"],
                     x_label="Run",
                     y_label="Network Usage (MB)")
        # bar chart for m bytes sent

    st.markdown("----")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.subheader("Cloud Avoidance Ratio")
        st.area_chart(dataframe.set_index("run_index")["cloud_avoidance_ratio"],
                      x_label="Run",
                      y_label="Avoidance Ratio")
    with c5:
        st.subheader("Cloud Infer Time")
        st.line_chart(dataframe.set_index("run_index")["cloud_infer_ms"],
                      x_label="Run",
                      y_label="Infer Time (ms)")
    with c6:
        st.subheader("Cloud Round-Trip Latency")
        st.line_chart(dataframe.set_index("run_index")["round_trip_ms"],
                      x_label="Run",
                      y_label="Round Trip (ms)")

    # RAW DISPLAY OF METRICS
    st.markdown("----")
    if show_raw:
        st.json(history[-1])  # print the raw JSON of the latest run
    st.subheader("Runs Table")
    st.dataframe(dataframe)
    #interactive table showing DataFrame

    if refresh_interval > 0:
        time.sleep(refresh_interval)
        # pause for the configured number of seconds
        st.rerun()
        # tell Streamlit to rerun the script for live refresh


if __name__ == "__main__":
    main()
