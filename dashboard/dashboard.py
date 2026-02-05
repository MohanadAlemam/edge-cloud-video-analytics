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
