## STEP 1: DISTRIBUTED DEPLOYMENT MODEL (Edge / Cloud / Dashboard)

create and install the following packages to their respective machines/VMs.

| Package | What to include | Purpose / Notes                                                                               |
|---------|-----------------|-----------------------------------------------------------------------------------------------|
| Cloud package | `cloud/`, `common/`, `requirements.txt` | Deploy to the cloud VM. Start `cloud/server.py`. `common/` must be present.                   |
| Edge package | `edge/`, `common/`, `data/`, `dashboard/`, `output/`, `requirements.txt` | Deploy to the edge VM. Run `edge/orchestrator.py`. This Writes `output/metrics_history.json`. |
| Dashboard | `dashboard/` | Can run on the edge VM or any machine with access to `output/metrics_history.json`.           |

### Deployment Notes

> **Note**
>
> - `common/` and `requirements.txt` have to be included in both Edge and Cloud packages.
> - `output/` is created automatically by the orchestrator if it does exist.
> - `experiments/` contains the Jupyter notebook and script and analysis used for performance evaluation.

---

## Running the System

### 1. Instantiate and activate a virtual environment (on both machines)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install dependencies (on both machines)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Start the Cloud Server (From the project root):

```bash
python -m cloud.server --port 5000
```

By default the server listens on all interfaces (`0.0.0.0`), awaiting edge requests.

---

### 4. Start the Dashboard

```bash
streamlit run dashboard/dashboard.py
```

The dashboard periodically and continuously reads this JSON file:

```text
output/metrics_history.json
```

---

### 5. Run the Edge Orchestrator

```bash
python -m edge.orchestrator \
    --video_path "data/experiment_sample.mp4" \
    --server_url "http://xxxxxxxxxxx.xxxx" 
```
Insert the Server URL  that is currently running.

Optional:

- Add `--store_video` to save the annotated output video.
- Change `--video_path` and `--server_url` to correspond to the video and the currently running server URL.

---

## Experiment Configurations

### Cloud-only (Force Cloud Processing)

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

### Edge-only (Force Edge Processing)

```bash
python -m edge.orchestrator \
    --video_path "data/experiment_sample.mp4" \
    --skip_interval 0 \
    --heuristic_threshold 5.0 \
    --edge_conf_threshold -1 \
    --m_output_dir "experiments/edge_only" \
    --server_url "http://10.0.0.3:5000"
```

For the Edge only experiment temporarily modify the decision gateway in `edge/edge_model.py`:

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

### Enable / Disable the Heuristic Filter

```text
--heuristic_threshold 5.0   # Default (enabled)

--heuristic_threshold 0     # Disabled
```

---

## Output Files

- `output/metrics_history.json` – Live edge cloud, network and scene metrics.
- `output/annotated_output.mp4` – Annotated output video (when `--store_video` is enabled).
- `experiments/experiments.ipynb` – Experimental analysis plots and observations.
