import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
from datetime import datetime

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Hybrid Bridge Inspector v6.8", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px; }
    
    .urgency-1 { background-color: #dc3545; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; }
    .urgency-2 { background-color: #fd7e14; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; }
    .urgency-3 { background-color: #28a745; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; }
    
    .arrow-box { font-size: 24px; text-align: center; margin: 5px 0; color: #6c757d; }
    
    /* Better Table Styling */
    th { background-color: #f1f3f5; }
</style>
""", unsafe_allow_html=True)

# --- 2. BRIDGE DATA SCHEMA ---
BRIDGE_SCHEMA = {
    "Superstructure": {
        "Deck": {"name_th": "พื้นสะพาน", "defects": ["Cracking", "Spalling", "Corrosion (Rebar)", "Wear/Abrasion", "Delamination"]},
        "Girder": {"name_th": "คานตามยาว", "defects": ["Flexure Cracks", "Shear Cracks", "Spalling", "Corrosion (Rebar)", "Excessive Deflection"]},
        "Diaphragm": {"name_th": "ค้ำยันคาน", "defects": ["Cracking", "Spalling"]},
        "Expansion Joint": {"name_th": "รอยต่อ", "defects": ["Clog", "Leakage", "Component Failure"]}
    },
    "Substructure": {
        "Cap Beam": {"name_th": "คานรัดหัวเสา", "defects": ["Cracking", "Corrosion (Rebar)", "Spalling"]},
        "Pier": {"name_th": "เสาตอม่อ", "defects": ["Settlement/Tilt", "Scour", "Spalling", "Cracking"]},
        "Footing": {"name_th": "ฐานราก", "defects": ["Scour/Exposure", "Settlement"]},
        "Bearing": {"name_th": "แผ่นรองรับคาน", "defects": ["Deformation", "Corrosion", "Slippage"]}
    }
}

# --- 3. HYBRID LOGIC ENGINE ---
def calculate_hybrid_assessment(defect_type, measured_val, component_name):
    doh_rating = 5 
    
    if "Crack" in defect_type:
        if measured_val > 0.005: doh_rating = 1      # > 5mm
        elif measured_val > 0.002: doh_rating = 2    # > 2mm
        elif measured_val > 0.0003: doh_rating = 3   # > 0.3mm
        elif measured_val > 0: doh_rating = 4        # < 0.3mm
    elif any(x in defect_type for x in ["Spalling", "Delamination", "Wear"]):
        if measured_val > 0.15: doh_rating = 1       # > 15cm
        elif measured_val > 0.10: doh_rating = 2     # > 10cm
        elif measured_val > 0.025: doh_rating = 3    # > 2.5cm
        elif measured_val > 0: doh_rating = 4
    elif "Corrosion" in defect_type:
        if measured_val > 0.30: doh_rating = 1       # > 30% Loss
        elif measured_val > 0