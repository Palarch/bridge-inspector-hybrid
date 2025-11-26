import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Hybrid Bridge Inspector (Complete)", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px;}
    
    /* Urgency Badges (ตามมาตรฐานรูปที่ 2) */
    .urgency-1 { background-color: #dc3545; color: white; padding: 5px 12px; border-radius: 15px; font-weight: bold; }
    .urgency-2 { background-color: #fd7e14; color: white; padding: 5px 12px; border-radius: 15px; font-weight: bold; }
    .urgency-3 { background-color: #28a745; color: white; padding: 5px 12px; border-radius: 15px; font-weight: bold; }
    
    /* Mapping Arrow */
    .arrow-box { font-size: 20px; text-align: center; margin: 10px 0; color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# --- 2. HYBRID LOGIC ENGINE (Standard Compliance) ---
def calculate_hybrid_assessment(defect_type, measured_depth, component_name):
    # STAGE 1: DOH Detection (5-0 Scale)
    doh_rating = 5 # Default Good
    
    if defect_type != "No Defect":
        if defect_type == "Spalling": # Threshold ตามรูป 4 (>20mm = Rating 1)
            if measured_depth > 0.020: doh_rating = 1
            elif measured_depth > 0.010: doh_rating = 2
            elif measured_depth > 0.005: doh_rating = 3
            else: doh_rating = 4
        elif defect_type == "Crack":
            if measured_depth > 0.005: doh_rating = 1
            elif measured_depth > 0.003: doh_rating = 2
            elif measured_depth > 0.001: doh_rating = 3
            else: doh_rating = 4
        elif defect_type == "Corrosion":
             if measured_depth > 0.05: doh_rating = 2
             else: doh_rating = 3

    # STAGE 2: Mapping (Invert Scale)
    mapping_table = {5:1, 4:2, 3:3, 2:4, 1:5, 0:5}
    cv_score = mapping_table.get(doh_rating, 1)

    # STAGE 3: Evaluation (Pellegrini)
    # Component Weight
    weight = 1.0
    if component_name in ["Girder", "Pier", "Cap Beam"]:
        weight = 1.5 # Primary Member
    
    priority_score = cv_score * weight
    
    # Action Plan
    if priority_score >= 6.0:
        return doh_rating, cv_score, weight, "Urgency 1 (High)", "ซ่อมทันที (Repair Immediately)", "urgency-1"
    elif priority_score >= 3.0:
        return doh_rating, cv_score, weight, "Urgency 2 (Medium)", "ซ่อมระยะสั้น (Short-term Repair)", "urgency-2"
    else:
        return doh_rating, cv_score, weight, "Urgency 3 (Low)", "เฝ้าระวัง (Monitor)", "urgency-3"

# --- 3. DATA HANDLER (Real File OR Mockup) ---
def get_bridge_data(uploaded_file, mock_item):
    if uploaded_file is not None:
        # A. FILE UPLOAD MODE
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower() for c in df.columns]
            if {'x','y','z'}.issubset(df.columns):
                if len(df) > 50000: df = df.sample(50000) # Downsample
                
                X, Y, Z = df['x'].values, df['y'].values, df['z'].values
                # Simple AI for file
                ai_d = abs(np.min(Z) - np.mean(Z)) if len(Z) > 0 else 0
                return X, Y, Z, ai_d, "Real File"
        except: pass

    # B. MOCKUP MODE (Detailed Structure v3.1)
    return generate_complex_structure(mock_item['type'], mock_item['comp'])

def generate_complex_structure(defect_type, component_name):
    points = []
    # Box Generator
    def add_box(xr, yr, zr, d=10):
        xx=np.linspace(xr[0],xr[1],int((xr[1]-xr[0])*d)); yy=np.linspace(yr[0],yr[1],int((yr[1]-yr[0])*d)); zz=np.linspace(zr[0],zr[1],4)
        X,Y=np.meshgrid(xx,yy); points.append(np.stack([X.flatten(),Y.flatten(),np.full_like(X,zr[0]).flatten()],axis=1))
        X,Y=np.meshgrid(xx,yy); points.append(np.stack([X.flatten(),Y.flatten(),np.full_like(X,zr[1]).flatten()],axis=1))
        X,Z=np.meshgrid(xx,zz); points.append(np.stack([X.flatten(),np.full_like(X,yr[0]).flatten(),Z.flatten()],axis=1))
        X,Z=np.meshgrid(xx,zz); points.append(np.stack([X.flatten(),np.full_like(X,yr[1]).flatten(),Z.flatten()],axis=1))

    # Dimensions
    L=12; W=8
    
    # Build Components
    add_box([0,L],[0,W],[0,0.2]) # Deck
    for y in [2,4,6]: add_box([0,L],[y-0.25,y+0.25],[-1.2,0]) # Girders
    for sx in [2,10]: 
        add_box([sx-0.4,sx+0.4],[0.5,7.5],[-2.0,-1.2]) # Cap
        for py in [2.5,5.5]: add_box([sx-0.3,sx+0.3],[py-0.3,py+0.3],[-5.0,-2.0]) # Pier

    full = np.concatenate(points, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    # Simulate Defect
    Z += np.random.normal(0, 0.003, size=Z.shape)
    ai_depth = 0.0
    
    if defect_type != "No Defect":
        mask = np.zeros_like(Z, dtype=bool)
        if component_name == "Deck":
            mask = (Z>-0.1)&((X-6)**2+(Y-4)**2<2)
            Z[mask] -= 0.025 # 25mm (>20mm Critical)
        elif component_name == "Girder":
            mask = (Z<-1.0)&(abs(Y-4)<0.3)&(abs(X-6)<0.3)
            Z[mask] += 0.006 # 6mm crack
        elif component_name == "Cap Beam":
            mask = (abs(X-2)<0.5)&(Z>-2.0)&(Y<2.0)
            Z[mask] -= 0.015 # 15mm
            
        if np.any(mask):
            ai_depth = abs(np.min(Z[mask])-np.mean(Z[~mask])) if defect_type=="Spalling" else np.max(Z[mask])-np.mean(Z[~mask])

    return X, Y, Z, ai_depth, "Mockup"

# --- 4. UI SETUP ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []

mock_data = [
    {"id":"C1", "comp":"Deck", "group":"Superstructure", "type":"Spalling", "loc":"Mid-span"},
    {"id":"C2", "comp":"Girder", "group":"Superstructure", "type":"Crack", "loc":"G2 Beam"},
    {"id":"C3", "comp":"Cap Beam", "group":"Substructure", "type":"Spalling", "loc":"Pier 1 Cap"},
]

# Sidebar
st.sidebar.title("🛠️ Tools")
uploaded_file = st.sidebar.file_uploader("Upload Point Cloud (.csv)", type=['csv'])
if st.session_state.results:
    df_ex = pd.DataFrame(st.session_state.results)
    st.sidebar.download_button("📥 Backup CSV", df_ex.to_csv(index=False).encode('utf-8'), "backup.csv", "text/csv")

# Main Page
st.title("🌉 Hybrid Bridge Inspector v5.0")
st.caption("Standards: DOH Detection ➔ Pellegrini Management | Feature: 3D/2D Analysis + File Support")

if st.session_state.idx >= len(mock_data):
    st.success("Inspection Batch Completed!")
    if st.session_state.results: st.dataframe(pd.DataFrame(st.session_state.results))
    if st.button("Restart"): st.session_state.idx=0; st.session_state.results=[]; st.rerun()
    st.stop()

# Load Data
item = mock_data[st.session_state.idx]
X, Y, Z, ai_depth, source = get_bridge_data(uploaded_file, item)

# Calculate Hybrid Rating
doh, cv, w, urgency, action, css = calculate_hybrid_assessment(item['type'], ai_depth, item['comp'])

# Layout
col_viz