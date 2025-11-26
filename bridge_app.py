import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Hybrid Bridge Inspector v5.3 (Stable)", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px; }
    
    .urgency-1 { background-color: #dc3545; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; }
    .urgency-2 { background-color: #fd7e14; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; }
    .urgency-3 { background-color: #28a745; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; }
    
    .arrow-box { font-size: 24px; text-align: center; margin: 5px 0; color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# --- 2. HYBRID LOGIC ENGINE ---
def calculate_hybrid_assessment(defect_type, measured_depth, component_name):
    doh_rating = 5 # Default
    
    if defect_type != "No Defect":
        if defect_type == "Spalling":
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

    mapping_table = {5:1, 4:2, 3:3, 2:4, 1:5, 0:5}
    cv_score = mapping_table.get(doh_rating, 1)

    weight = 1.0
    if component_name in ["Girder", "Pier", "Cap Beam"]:
        weight = 1.5 
    
    priority_score = cv_score * weight
    
    if priority_score >= 6.0:
        return doh_rating, cv_score, weight, "Urgency 1 (High)", "ซ่อมทันที (Repair Immediately)", "urgency-1"
    elif priority_score >= 3.0:
        return doh_rating, cv_score, weight, "Urgency 2 (Medium)", "ซ่อมระยะสั้น (Short-term Repair)", "urgency-2"
    else:
        return doh_rating, cv_score, weight, "Urgency 3 (Low)", "เฝ้าระวัง (Monitor)", "urgency-3"

# --- 3. OPTIMIZED STRUCTURE GENERATOR (Lite Version) ---
def generate_complex_structure(defect_type, component_name):
    points_list = []
    
    # Adjusted Density: ลดจำนวนจุดลงเพื่อป้องกัน Web Crash
    def add_dense_block(x_lim, y_lim, z_lim, density=400): 
        vol = (x_lim[1]-x_lim[0]) * (y_lim[1]-y_lim[0]) * (z_lim[1]-z_lim[0])
        n_points = int(density * vol)
        
        # Safety Cap: จำกัดจุดต่อชิ้นไม่ให้เกิน 5000
        if n_points > 5000: n_points = 5000
        if n_points < 200: n_points = 200
        
        # Random Points
        xx = np.random.uniform(x_lim[0], x_lim[1], n_points)
        yy = np.random.uniform(y_lim[0], y_lim[1], n_points)
        zz = np.random.uniform(z_lim[0], z_lim[1], n_points)
        
        # Surface (Wireframe-like)
        xe = np.linspace(x_lim[0], x_lim[1], 10)
        ye = np.linspace(y_lim[0], y_lim[1], 10)
        Xg, Yg = np.meshgrid(xe, ye)
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[0]).flatten()], axis=1)) 
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[1]).flatten()], axis=1)) 
        
        points_list.append(np.stack([xx, yy, zz], axis=1))

    # Dimensions
    L = 12.0; W = 8.0
    z_deck_bot = -0.3; z_girder_bot = -1.5; z_cap_bot = -2.5; z_pier_bot = -6.0

    # Build Bridge (Reduced Density)
    add_dense_block([0, L], [0, W], [z_deck_bot, 0], density=600) # Deck
    
    for y in [2.0, 4.0, 6.0]:
        add_dense_block([0, L], [y-0.3, y+0.3], [z_girder_bot, z_deck_bot], density=500) # Girders
        
    for x in [0.5, L/2, L-0.5]:
        add_dense_block([x-0.15, x+0.15], [2.0, 4.0], [z_girder_bot+0.3, z_deck_bot-0.1], density=300)
        add_dense_block([x-0.15, x+0.15], [4.0, 6.0], [z_girder_bot+0.3, z_deck_bot-0.1], density=300)

    for sx in [2.0, 10.0]:
        add_dense_block([sx-0.6, sx+0.6], [0.5, W-0.5], [z_cap_bot, -1.5], density=600) # Cap
        for py in [2.5, 5.5]:
            add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_pier_bot, z_cap_bot], density=500) # Pier

    full = np.concatenate(points_list, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    # Simulate Defect
    Z += np.random.normal(0, 0.005, size=Z.shape)
    ai_depth = 0.0
    
    if defect_type != "No Defect":
        mask = np.zeros_like(Z, dtype=bool)
        if component_name == "Deck" and defect_type == "Spalling":
            mask = (Z > -0.1) & ((X-6)**2 + (Y-4)**2 < 2.5)
            Z[mask] -= 0.15
        elif component_name == "Girder" and defect_type == "Crack":
            mask = (Z < z_girder_bot+0.5) & (abs(Y-4.0)<0.35) & (abs(X-6.0)<0.2)
            Z[mask] += 0.08
        elif component_name == "Cap Beam" and defect_type == "Spalling":
            mask = (abs(X-2.0)<0.7) & (Z>z_cap_bot) & (Y<2.0)
            Z[mask] -= 0.025
            
        if np.any(mask):
            if defect_type == "Spalling": ai_depth = abs(np.min(Z[mask]) - np.mean(Z[~mask]))
            else: ai_depth = np.max(Z[mask]) - np.mean(Z[~mask])

    return X, Y, Z, ai_depth, "Mockup (Lite)"

# --- 4. DATA HANDLER ---
def get_bridge_data(uploaded_file, mock_item):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower() for c in df.columns]
            if {'x','y','z'}.issubset(df.columns):
                # Aggressive Downsampling for real files
                if len(df) > 20000: df = df.sample(20000)
                X, Y, Z = df['x'].values, df['y'].values, df['z'].values
                ai_d = abs(np.min(Z) - np.mean(Z)) if len(Z)>0 else 0
                return X, Y, Z, ai_d, "Real File"
        except: pass
    return generate_complex_structure(mock_item['type'], mock_item['comp'])

# --- 5. MAIN APP ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []

mock_data = [
    {"id":"01", "comp":"Deck", "group":"Superstructure", "type":"Spalling", "loc":"Mid-span Lane 1"},
    {"id":"02", "comp":"Girder", "group":"Superstructure", "type":"Crack", "loc":"G2 Mid-span"},
    {"id":"03", "comp":"Cap Beam", "group":"Substructure", "type":"Spalling", "loc":"Pier Cap Left"},
    {"id":"04", "comp":"Pier", "group":"Substructure", "type":"No Defect", "loc":"Pier 1"},
]

st.sidebar.title("🛠️ Tools")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if st.session_state.results:
    st.sidebar.download_button("📥 Backup Data", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")

st.title("🌉 Hybrid Bridge Inspector v5.3 (Stable)")
st.caption("Standard: DOH Detection ➔ Pellegrini Management")

if st.session_state.idx >= len(mock_data):
    st.success("✅ Inspection Completed!")
    st.dataframe(pd.DataFrame(st.session_state.results))
    if st.button("Restart"): st.session_state.idx=0; st.session_state.results=[]; st.rerun()
    st.stop()

item = mock_data[st.session_state.idx]
X, Y, Z, ai_depth, source = get_bridge_data(uploaded_file, item)
doh, cv, w, urgency, action, css = calculate_hybrid_assessment(item['type'], ai_depth, item['comp'])

# Layout
col_viz, col_data = st.columns([1.8, 1])

with col_viz:
    st.subheader(f"📍 {item['comp']} ({source})")
    
    # 1. SLIDER & 2D GRAPH
    st.markdown("##### ✂️ Cross-Section Analyzer")
    if len(X) > 0:
        slice_pos = st.slider("X-Axis Cut", float(np.min(X)), float(np.max(X)), float(