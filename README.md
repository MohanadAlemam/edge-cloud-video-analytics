## DISTRIBUTED EDGE–CLOUD AI PIPELINE FOR REAL-TIME VIDEO ANALYTICS

### 1. PROBLEM OVERVIEW

Designing and implementing efficient real-time edge-cloud video analytics AI solutions is a multi-objective optimization problem, which requires an intricate balancing and trade-off between latency, privacy, bandwidth, and accuracy for the specific target application.

This project is a ***prototype*** of an edge-cloud video analytics distributed system that prioritizes efficiency and demonstrates the trade-off between cost (i.e., low latency and bandwidth preservation) and quality/accuracy of object detection.

### 2. KEY FEATURES

This prototype features an intelligent edge device equipped with a heuristic frame filter and on-edge lightweight object detection model to reduce reliance on the server. The pipeline performs selective cloud offloading, and the live dashboard displays key performance metrics in near-real time. Furthermore, the system features live displays of the annotated video for monitoring and live observability.

### 3. LIVE DEMONSTRATION

#### 3.1 Live annotated video

![Demo GIF 1](images/gif_annotated_dmo.gif)

#### 3.2 Live Streamlit dashboard

![Demo GIF 2](images/gif_dashboard.gif)

### 4. MAIN DESIGN OBJECTIVES

- Low latency near-real-time object detection.
- Bandwidth efficiency by minimizing cloud/server reliance.
- Good object detection quality/accuracy.
- Provide high monitoring and observability interface for operator's awareness.

### 5. ARCHITECTURE AND WORKFLOW

This prototype adopts an edge-cloud architecture (i.e. Edge - Cloud - Dashboard). The edge decodes the video, pre-processes frames, and deploys a heuristic motion detector to filter out uninteresting frames. A lightweight on-edge model conducts inference and selectively offloads frames to the cloud. The cloud performs heavy-weight object detection and returns results to the edge. Concurrently, a Streamlit dashboard reads and displays live metrics.

![Archetechure](images/architechture.png)

### 6. KEY TECH STACK

The core building-block packages and tools include Python, Ultralytics (YOLOv8), PyTorch, OpenCV, Flask, and Streamlit. Refer to requirements.txt for the detailed list of dependencies.

### 7. KEY INSIGHTS AND RESULTS

System-level experiments were conducted to assess the impact of system modules/components on the overall pipeline efficiency. Results were analyzed showing the following key insights:
- Object detection quality/accuracy: Utilizing the lightweight edge model and selectively offloading frames to the cloud produces a marginally lower but comparable detection quality/accuracy, and gives a significant boost in detection speed.
- Latency reduction: The edge heuristic filter and lightweight model (i.e., edge intelligence) significantly reduced overall latency (edge inference is approximately 2× faster than cloud).

![Prediction quality](images/smoothed_per_frame_time_series.png) 

- Monitoring and observability: The dashboard provides live display of key performance indicators (e.g., latency and bandwidth) and scene analysis (events and objects detected).

- Bandwidth savings: With the heuristic filter set to default, edge intelligence resulted in substantial bandwidth savings (approximately 70%).

![Heuristic Filter Impact](images/heuristic_filter_ON_OFF.png)

### 8. Technical Report 

For detailed information on the system's design, implementation, and evaluation, refer to the technical report at *docs/Technical_Report.pdf*.

### 9. PROJECT STRUCTURE AND GUIDE

```
cloud-edge-video-analytics/     <- the project root

├─ cloud/
│  ├─ server.py                 # main Flask cloud server: POST /infer
│  └─ utilities.py              # helper functions
│
├─ common/
│  ├─ frame_content.py          # application level code: intrusion detection
│  ├─ metrics_snapshot.py       # write_metrics_snapshot() to output/metrics_history.json
│  └─ visualize.py              # parse_detections() and annotate_frame()
│
├─ dashboard/
│  └─ dashboard.py              # this is a streamlit app reads output/metrics_history.json
│
├─ data/
│  ├─ experiment_sample.mp4     # video samples used in the experiments and demo
│  ├─ demo1_1080_30fps.mp4      # additional demo video
│  └─ demo3_1080_30fps.mp4      # project demo video
│
├─ docs/
│  ├─ Technical_Report.pdf   # report detailed system design implementation and evaluation
│  ├─ CLI_INTERFACE.md          # CLI flags and reference
│  └─ DEPLOYMENT.md           # deployment instructions
│
├─ edge/
│  ├─ orchestrator.py        # orchestrator and main CLI entry interface
│  ├─ cloud_feeder.py           # feed_cloud_jpeg() communicator
│  ├─ edge_model.py          # EdgeModel class: includes lazy import, warmup and lock
│  ├─ preprocess.py             # for heuristic filtering, frame resizing, and converting forms to grayscale
│  └─ video_reader.py        # breaks down video stream to individual frames
│
├─ experiments/
│  ├─ e_utilities.py         # helper analysis functions
│  └─ experiments.ipynb       # experiments notebook: load JSONs, plots, figures and observations
│
├─ images/              # figures and diagrams and demo GIFs used in README
│
├─ output/
│  ├─ metrics_history.json      # generated by edge to save metrics snapshots
│  └─ annotated_output.mp4   # if activated this is where the generated annotated video output
│
├─ .gitignore
├─ requirements.txt
├─ LICENSE
└─ README.md

```



### 10. Quick Start

#### 10.1. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
````

---

#### 10.2. Run the system (from project root)

#### Start Cloud Server

```bash
python -m cloud.server --port 5000
```

#### Start Dashboard

```bash
streamlit run dashboard/dashboard.py
```

#### Start Edge Orchestrator

```bash
python -m edge.orchestrator --video_path "data/demo3_1080_30fps.mp4" --server_url "http://127.0.0.1:5000"
```

---

#### NOTE

Amend CLI flags `--video_path` and `--server_url` to correspond to the video and the currently running server URL.



