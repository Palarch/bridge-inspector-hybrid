import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
from datetime import datetime

# --- 1. CONFIG & CSS SETUP ---
st.set_page_config(page_title="Hybrid Bridge Inspector v6.6", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px; }
    
    .urgency-1 { background-color: #dc3545; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    .urgency-2 { background-color: #fd7e14; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    .urgency-3 { background-color: #28a745; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    
    .arrow-box { font-size: 24px; text-align: center; margin: 5px 0; color: #6c757d; }
    
    thead tr th:first-child { display:none }
    tbody th { display:none }
</style>
""", unsafe_allow_html=True)

# --- 2. BRIDGE DATA SCHEMA ---
BRIDGE_SCHEMA = {
    "Superstructure": {
        "Deck": {"name_th": "พื้นสะพาน", "defects": ["Cracking", "Spalling", "Corrosion (Rebar)", "Wear/Abrasion"]},
        "Girder": {"name_th": "คานตามยาว", "defects": ["Flexure Cracks", "Shear Cracks", "Spalling", "Corrosion (Rebar)", "Excessive Deflection"]},
        "Diaphragm": {"name_th": "ค้ำยันคาน", "defects": ["Cracking", "Spalling"]},
    },
    "Substructure": {
        "Cap Beam": {"name_th": "คานรัดหัวเสา", "defects": ["Cracking", "Corrosion (Rebar)", "Spalling"]},
        "Pier": {"name_th": "เสาตอม่อ", "defects": ["Settlement/Tilt", "Scour", "Spalling", "Cracking"]},
        "Footing": {"name_th": "ฐานราก", "defects": ["Scour/Exposure", "Settlement"]},
        "Bearing": {"name_th": "แผ่นรองรับคาน", "defects": ["Deformation", "Corrosion", "Slippage"]}
    }
}

# --- 3. HYBRID LOGIC ENGINE (Updated with ALL Severity Types) ---
def calculate_hybrid_assessment(defect_type, measured_val, component_name):
    # STAGE 1: DOH DETECTION (5-0 Scale)
    doh_rating = 5 # Default Good
    
    # 1. Cracking (วัดความกว้าง mm)
    if "Crack" in defect_type:
        if measured_val > 0.005: doh_rating = 1      # > 5 mm (Critical)
        elif measured_val > 0.002: doh_rating = 2    # > 2 mm (Serious)
        elif measured_val > 0.0003: doh_rating = 3   # > 0.3 mm (Poor)
        elif measured_val > 0: doh_rating = 4        # < 0.3 mm (Fair)
        
    # 2. Spalling / Void (วัดความลึก/ขนาด m)
    elif "Spalling" in defect_type or "Void" in defect_type:
        if measured_val > 0.15: doh_rating = 1       # > 15 cm (Critical)
        elif measured_val > 0.10: doh_rating = 2     # > 10 cm (Serious)
        elif measured_val > 0.025: doh_rating = 3    # > 2.5 cm (Poor)
        elif measured_val > 0: doh_rating = 4        # (Fair)
        
    # 3. Corrosion (วัด % การสูญเสียหน้าตัดเหล็ก หรือ พื้นที่สนิม)
    elif "Corrosion" in defect_type:
        if measured_val > 0.30: doh_rating = 1       # > 30% Loss (Critical - เหล็กขาด)
        elif measured_val > 0.10: doh_rating = 2     # > 10% Loss (Serious)
        elif measured_val > 0.01: doh_rating = 3     # Pitting/Surface (Poor)
        elif measured_val > 0: doh_rating = 4        # Light Rust (Fair)
        
    # 4. Scour / Settlement / Tilt (วัดการเคลื่อนตัว m)
    elif "Scour" in defect_type or "Settlement" in defect_type or "Tilt" in defect_type:
        if measured_val > 0.50: doh_rating = 1       # > 50 cm (Critical - ฐานรากลอย)
        elif measured_val > 0.20: doh_rating = 2     # > 20 cm (Serious)
        elif measured_val > 0.05: doh_rating = 3     # > 5 cm (Poor)
        elif measured_val > 0: doh_rating = 4        # (Fair)

    if defect_type == "No Defect": doh_rating = 5

    # STAGE 2: MAPPING (Invert Scale)
    mapping_table = {5:1, 4:2, 3:3, 2:4, 1:5, 0:5}
    cv_score = mapping_table.get(doh_rating, 1)

    # STAGE 3: EVALUATION (Pellegrini)
    weight = 1.0
    primary_comps = ["Girder", "Pier", "Cap Beam", "Footing", "Bearing"]
    if any(p in component_name for p in primary_comps):
        weight = 1.5
    
    priority_score = cv_score * weight
    
    # Urgency
    if priority_score >= 6.0:
        return doh_rating, cv_score, weight, "Urgency 1 (High)", "ซ่อมทันที (Repair Immediately)", "urgency-1"
    elif priority_score >= 3.0:
        return doh_rating, cv_score, weight, "Urgency 2 (Medium)", "ซ่อมระยะสั้น (Short-term Repair)", "urgency-2"
    else:
        return doh_rating, cv_score, weight, "Urgency 3 (Low)", "เฝ้าระวัง (Monitor)", "urgency-3"

# --- 4. STRUCTURE GENERATOR ---
def generate_complex_structure(defect_type, component_name):
    points_list = []
    
    def add_dense_block(x_lim, y_lim, z_lim, density=400): 
        vol = (x_lim[1]-x_lim[0]) * (y_lim[1]-y_lim[0]) * (z_lim[1]-z_lim[0])
        n_points = int(density * vol)
        if n_points > 4000: n_points = 4000
        if n_points < 200: n_points = 200
        
        xx = np.random.uniform(x_lim[0], x_lim[1], n_points)
        yy = np.random.uniform(y_lim[0], y_lim[1], n_points)
        zz = np.random.uniform(z_lim[0], z_lim[1], n_points)
        
        xe = np.linspace(x_lim[0], x_lim[1], 10)
        ye = np.linspace(y_lim[0], y_lim[1], 10)
        Xg, Yg = np.meshgrid(xe, ye)
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[0]).flatten()], axis=1))
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[1]).flatten()], axis=1))
        points_list.append(np.stack([xx, yy, zz], axis=1))

    L = 12.0; W = 8.0
    z_deck_bot = -0.3; z_girder_bot = -1.5; z_cap_bot = -2.5; z_pier_bot = -6.0; z_foot_bot = -7.0

    add_dense_block([0, L], [0, W], [z_deck_bot, 0], density=600)
    for y in [2.0, 4.0, 6.0]: add_dense_block([0, L], [y-0.3, y+0.3], [z_girder_bot, z_deck_bot], density=500)
    for sx in [2.0, 10.0]:
        add_dense_block([sx-0.6, sx+0.6], [0.5, W-0.5], [z_cap_bot, -1.5], density=600)
        for py in [2.5, 5.5]: 
            add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_pier_bot, z_cap_bot], density=500)

    full = np.concatenate(points_list, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    Z += np.random.normal(0, 0.005, size=Z.shape)
    ai_depth = 0.0
    
    if defect_type != "No Defect":
        mask = np.zeros_like(Z, dtype=bool)
        if "Deck" in component_name: mask = (Z > -0.1) & ((X-6)**2 + (Y-4)**2 < 2.5)
        elif "Girder" in component_name: mask = (Z < z_girder_bot+0.5) & (abs(Y-4.0)<0.4) & (abs(X-6.0)<0.3)
        elif "Cap" in component_name: mask = (abs(X-2.0)<0.7) & (Z>z_cap_bot) & (Y<2.0)
        elif "Pier" in component_name: mask = (abs(X-2.0)<0.5) & (Z<z_cap_bot-1) & (Y<3.0)
        
        if np.any(mask):
            severity_factor = 0.02
            if "Crack" in defect_type: severity_factor = 0.005
            elif "Spalling" in defect_type: severity_factor = 0.15
            elif "Scour" in defect_type: severity_factor = 0.30
            elif "Corrosion" in defect_type: severity_factor = 0.05
            
            Z[mask] -= severity_factor
            ai_depth = severity_factor

    return X, Y, Z, ai_depth

# --- 5. DATA GENERATOR ---
def generate_