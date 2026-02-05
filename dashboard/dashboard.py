#-----------------------------------------------------------------------------------------------------------------------
# EDGE CLOUD VIDEO ANALYTICS METRICS DASHBOARD
#-----------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import json

import plotly.express as px
import time
from pathlib import Path


METRICS_PATH = Path("output/metrics_log.json")
# path to metrics file produced by orchestrator

st.set_page_config(page_title="Edge-Cloud Video Analytics Dashboard", layout="wide")
st.title("Edge-Cloud Video Analytics - Metrics Dashboard")
# main title on the dashboard

st.sidebar.header("Settings")
refresh_interval = st.sidebar.number_input(
    "Auto-refresh interval (seconds, 0 = off)",
    min_value=0,
    value=5,
    step=1
)

show_raw =st.sidebar.checkbox("Show raw JSON", value=False)
# checkbox to toggle showing raw JSON
max_runs = st.sidebar.number_input("Max runs shown",min_value=1, value=30, step=1)
# how many recent runs to show
