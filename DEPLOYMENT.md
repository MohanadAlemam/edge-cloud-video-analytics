# STEP 1: DISTRIBUTED DEPLOYMENT MODEL (Edge / Cloud / Dashboard)

create and install the following packages to their respective machines/VMs.

| Package | What to include | Purpose / Notes |
|---------|-----------------|-----------------|
| Cloud package | `cloud/`, `common/`, `requirements.txt` | Deploy on the cloud VM. Start `cloud/server.py`. `common/` must be present. |
| Edge package | `edge/`, `common/`, `data/`, `dashboard/`, `output/`, `requirements.txt` | Deploy on the edge VM. Run `edge/orchestrator.py`. Writes `output/metrics_history.json`. |
| Dashboard | `dashboard/` | Can run on the edge VM or any machine with access to `output/metrics_history.json`. |

## Deployment Notes

> **Note**
>
> - `common/` and `requirements.txt` must be included in both the Edge and Cloud packages.
> - `output/` is created automatically by the orchestrator if it does not already exist.
> - `experiments/` contains the notebooks and scripts used for performance evaluation and analysis.

---

# Running the System

## 1. Create and activate a virtual environment (both machines)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 2. Install dependencies (both machines)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Start the Cloud Server

From the project root:

```bash
python -m cloud.server --port 5000
```

By default, the server listens on all interfaces (`0.0.0.0`), allowing edge devices to connect.

---

## 4. Start the Dashboard

```bash
streamlit run dashboard/dashboard.py
```

The dashboard monitors:

```text
output/metrics_history.json
```

---

## 5. Run the Edge Orchestrator

```bash
python -m edge.orchestrator \
    --video_path "data/experiment_sample.mp4" \
    --server_url "http://127.0.0.1:5000"
```

Optional:

- Add `--store_video` to save the annotated output video.
- Change `--server_url` when connecting to a remote cloud server.

---

# Experiment Configurations

## Cloud-only (Force Cloud Processing)

```bash
python -m edge.orchestrator \
    --video_path "data/experiment_sample.mp4" \
    --skip_interval 0 \
    --heuristic_threshold 5.0 \
    --edge_conf_threshold 1.1 \
    --m_output_dir "experiments/cloud_only" \
    --server_url "http://10.0.0.3:5000"
```

---

## Edge-only (Force Edge Processing)

```bash
python -m edge.orchestrator \
    --video_path "data/experiment_sample.mp4" \
    --skip_interval 0 \
    --heuristic_threshold 5.0 \
    --edge_conf_threshold -1 \
    --m_output_dir "experiments/edge_only" \
    --server_url "http://10.0.0.3:5000"
```

For the Edge-only experiment, temporarily modify the decision gateway in `edge/edge_model.py`:

Replace:

```python
if not confidences_list:
    return True
```

with

```python
if not confidences_list:
    return False
```

Restore the original logic after completing the experiment.

---

## Enable / Disable the Heuristic Filter

```text
--heuristic_threshold 5.0   # Default (enabled)

--heuristic_threshold 0     # Disabled
```

---

# Output Files

- `output/metrics_history.json` – Live edge, cloud, network and scene metrics.
- `output/annotated_output.mp4` – Annotated output video (when `--store_video` is enabled).
- `experiments/experiments.ipynb` – Experimental analysis, plots and observations.
