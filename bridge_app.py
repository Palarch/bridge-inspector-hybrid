import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
from datetime import datetime

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Hybrid Bridge Inspector v6.1 (Master)", layout="wide")

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

# --- 2. BRIDGE DATA SCHEMA (Full Database) ---
BRIDGE_SCHEMA = {
    "Superstructure": {
        "Deck": {"name_th": "พื้นสะพาน", "defects": ["Cracking", "Scaling", "Delamination", "Spalling", "Efflorescence", "Honeycomb", "Reinforcement Corrosion", "Pop-out", "Wear", "Collision Damage"]},
        "Girder": {"name_th": "คานตามยาว", "defects": ["Flexure Cracks", "Shear Cracks", "Excessive Deflection", "Spalling", "Collision Damage", "Honeycomb", "Reinforcement Corrosion"]},
        "Diaphragm": {"name_th": "ค้ำยันคานตามยาว", "defects": ["Cracking", "Abrasion", "Delamination"]},
        "Wearing Surface": {"name_th": "ผิวทาง", "defects": ["Potholing", "Rutting", "Map Cracking", "Delamination"]},
        "Expansion Joint": {"name_th": "รอยต่อเผื่อการขยาย", "defects": ["Clog", "Loss of Parallelism", "Trapped Water/Leakage", "Component Failure"]},
        "Railing & Barrier": {"name_th": "ราวสะพานและแผงกั้น", "defects": ["Collision Damage", "Spalling", "Reinforcement Corrosion"]}
    },
    "Substructure": {
        "Cap Beam": {"name_th": "คานรัดหัวเสา", "defects": ["Cracking", "Reinforcement Corrosion", "Efflorescence", "Honeycomb"]},
        "Pier & Pier Wall": {"name_th": "เสาตอม่อและกำแพง", "defects": ["Vertical Movement", "Lateral Movement/Tilting", "Scour", "Spalling", "Cracking"]},
        "Footing": {"name_th": "ฐานราก", "defects": ["Scour/Exposure", "Settlement"]},
        "Bearing": {"name_th": "แผ่นรองรับคาน", "defects": ["Shear Deformation", "Slippage", "Bulging", "Corrosion"]},
        "Bracing": {"name_th": "ค้ำยันตอม่อ", "defects": ["Reduce Cross-Section", "Collision Damage"]},
        "Secondary Substructure": {"name_th": "ส่วนประกอบรอง", "defects": ["Settlement/Tilt", "Scour/Void", "Cracking"]}
    }
}

# --- 3. HYBRID LOGIC ENGINE (DOH -> Mapping -> Pellegrini) ---
def calculate_hybrid_assessment(defect_type, measured_val, component_name):
    # STAGE 1: DOH DETECTION (5-0 Scale)
    doh_rating = 5
    
    # Generic Severity Logic
    if "Crack" in defect_type:
        if measured_val > 0.005: doh_rating = 1      # >5mm Critical
        elif measured_val > 0.002: doh_rating = 2    # >2mm Serious
        elif measured_val > 0.0005: doh_rating = 3   # >0.5mm Poor
        elif measured_val > 0: doh_rating = 4
    elif any(x in defect_type for x in ["Spalling", "Scaling", "Honeycomb", "Potholing", "Scour"]):
        if measured_val > 0.15: doh_rating = 1       # >15cm Critical
        elif measured_val > 0.10: doh_rating = 2
        elif measured_val > 0.05: doh_rating = 3
        elif measured_val > 0: doh_rating = 4
    elif any(x in defect_type for x in ["Movement", "Settlement", "Deflection", "Tilt"]):
        if measured_val > 0.05: doh_rating = 1
        elif measured_val > 0.025: doh_rating = 2
        elif measured_val > 0.010: doh_rating = 3
        elif measured_val > 0: doh_rating = 4
    else: 
        if measured_val > 0.1: doh_rating = 2
        elif measured_val > 0: doh_rating = 3

    if defect_type == "No Defect": doh_rating = 5

    # STAGE 2: MAPPING (Invert Scale)
    mapping_table = {5:1, 4:2, 3:3, 2:4, 1:5, 0:5}
    cv_score = mapping_table.get(doh_rating, 1)

    # STAGE 3: EVALUATION (Component Weighting)
    weight = 1.0
    primary_comps = ["Girder", "Pier & Pier Wall", "Cap Beam", "Footing", "Bearing"]
    if any(p in component_name for p in primary_comps):
        weight = 1.5
    
    priority_score = cv_score * weight
    
    if priority_score >= 6.0:
        return doh_rating, cv_score, weight, "Urgency 1 (High)", "ซ่อมทันที (Repair Immediately)", "urgency-1"
    elif priority_score >= 3.0:
        return doh_rating, cv_score, weight, "Urgency 2 (Medium)", "ซ่อมระยะสั้น (Short-term Repair)", "urgency-2"
    else:
        return doh_rating, cv_score, weight, "Urgency 3 (Low)", "เฝ้าระวัง (Monitor)", "urgency-3"

# --- 4. STRUCTURE GENERATOR (Mockup) ---
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
        points_list.append(np.stack([xx, yy, zz], axis=1))

    L = 12.0; W = 8.0
    z_deck_bot=-0.3; z_girder_bot=-1.5; z_cap_bot=-2.5; z_pier_bot=-6.0
    
    # Build Geometry
    add_dense_block([0, L], [0, W], [z_deck_bot, 0], density=600) # Deck
    for y in [2.0, 4.0, 6.0]: add_dense_block([0, L], [y-0.3, y+0.3], [z_girder_bot, z_deck_bot], density=500) # Girder
    for sx in [2.0, 10.0]:
        add_dense_block([sx-0.6, sx+0.6], [0.5, W-0.5], [z_cap_bot, -1.5], density=600) # Cap
        for py in [2.5, 5.5]: add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_pier_bot, z_cap_bot], density=500) # Pier

    full = np.concatenate(points_list, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    # Simulate Defect
    Z += np.random.normal(0, 0.005, size=Z.shape)
    ai_depth = 0.0
    
    mask = np.zeros_like(Z, dtype=bool)
    if defect_type != "No Defect":
        if "Deck" in component_name: mask = (Z > -0.1) & ((X-6)**2 + (Y-4)**2 < 2.5)
        elif "Girder" in component_name: mask = (Z < z_girder_bot+0.5) & (abs(Y-4.0)<0.4) & (abs(X-6.0)<0.3)
        elif "Cap" in component_name: mask = (abs(X-2.0)<0.7) & (Z>z_cap_bot) & (Y<2.0)
        elif "Pier" in component_name: mask = (abs(X-2.0)<0.5) & (Z<z_cap_bot-1) & (Y<3.0)
        
        if np.any(mask):
            severity_factor = 0.005 if "Crack" in defect_type else 0.15
            Z[mask] -= severity_factor
            ai_depth = severity_factor

    return X, Y, Z, ai_depth, "Mockup Data"

# --- 5. DATA HANDLER (REAL FILE SUPPORT RESTORED!) ---
def get_inspection_data(uploaded_file, mock_item):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower() for c in df.columns]
            if {'x','y','z'}.issubset(df.columns):
                if len(df) > 30000: df = df.sample(30000)
                X, Y, Z = df['x'].values, df['y'].values, df['z'].values
                ai_d = abs(np.min(Z) - np.mean(Z)) if len(Z)>0 else 0
                return X, Y, Z, ai_d, "Real File Uploaded"
        except: pass
    
    # Fallback to Mockup
    return generate_complex_structure(mock_item['type'], mock_item['comp'])

# --- 6. DATA GENERATOR (Schema-based) ---
def generate_mock_batch():
    batch = []
    for _ in range(5):
        group = random.choice(list(BRIDGE_SCHEMA.keys()))
        comp = random.choice(list(BRIDGE_SCHEMA[group].keys()))
        defect = random.choice(BRIDGE_SCHEMA[group][comp]["defects"])
        depth = random.uniform(0.001, 0.008) if "Crack" in defect else random.uniform(0.02, 0.15)
        batch.append({
            "id": f"INS-{random.randint(100,999)}",
            "group": group, "comp": comp,
            "comp_th": BRIDGE_SCHEMA[group][comp]["name_th"],
            "type": defect, "depth": depth
        })
    return batch

# --- 7. MAIN UI ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []
if 'mock_data' not in st.session_state: st.session_state.mock_data = generate_mock_batch()

st.sidebar.title("🛠️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Point Cloud (.csv)", type=['csv'])
if st.sidebar.button("🔄 New Batch"):
    st.session_state.mock_data = generate_mock_batch()
    st.session_state.idx = 0; st.rerun()
if st.session_state.results:
    st.sidebar.download_button("📥 Export Report", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")

st.title("🌉 Hybrid Bridge Inspector v6.1 (Master)")
st.caption("Full Features: DOH/Pellegrini Standard + Real File Support + Smart Schema")

if st.session_state.idx >= len(st.session_state.mock_data):
    st.success("✅ Batch Completed!"); st.dataframe(pd.DataFrame(st.session_state.results))
    if st.button("Start Over"): st.session_state.idx=0; st.session_state.results=[]; st.rerun()
    st.stop()

# Processing
item = st.session_state.mock_data[st.session_state.idx]
# Decide source: Uploaded File OR Generated Mockup
X, Y, Z, ai_depth, source_txt = get_inspection_data(uploaded_file, item)
# If real file, use simple depth calc, otherwise use pre-gen
if "Real" not in source_txt: ai_depth = item['depth']

doh, cv, w, urgency, action, css = calculate_hybrid_assessment(item['type'], ai_depth, item['comp'])

# Layout
col_viz, col_data = st.columns([1.8, 1])

with col_viz:
    st.subheader(f"📍 {item['comp']} ({item['comp_th']})")
    st.caption(f"Defect: {item['type']} | Source: {source_txt}")
    
    # 1. SLIDER & 2D
    st.markdown("##### ✂️ Cross-Section Analyzer")
    if len(X) > 0:
        slice_pos = st.slider("X-Axis Cut", float(np.min(X)), float(np.max(X)), float(np.mean(X)))
        mask = np.abs(X - slice_pos) < 0.2
        
        fig_sec = go.Figure(go.Scatter(x=Y[mask], y=Z[mask], mode='markers', marker=dict(size=5, color=Z[mask], colorscale='Jet_r', opacity=0.8)))
        fig_sec.update_layout(template='plotly_white', height=250, title=f"Section at X={slice_pos:.1f}m", margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_sec, use_container_width=True)

        # 2. 3D
        fig_3d = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=3, color=Z, colorscale='Jet_r', opacity=0.7))])
        py, pz = np.meshgrid(np.linspace(np.min(Y), np.max(Y), 10), np.linspace(np.min(Z), np.max(Z), 10))
        px = np.full_like(py, slice_pos)
        fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
        fig_3d.update_layout(template='plotly_white', height=500, scene=dict(aspectmode='data'), margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.error("No Data Points")

with col_data:
    st.markdown("### 📊 Assessment Card")
    st.markdown(f"""
    <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
        <small>Stage 1: Detection (DOH)</small><br>
        Severity: <b>{ai_depth:.4f} m</b><br>Rating: <b>{doh}/5</b>
    </div>
    <div class="arrow-box">⬇️</div>
    <div style="border:1px solid #000080; padding:10px; border-radius:5px; background-color:#f0f4ff;">
        <small>Stage 3: Management</small><br>
        CV Score: <b>{cv}</b> (Weight x{w})<br>Priority Score: <b>{cv*w:.1f}</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""<div style="margin-top:15px; text-align:center;"><span class="{css}">{urgency}</span><br><h4>{action}</h4></div>""", unsafe_allow_html=True)
    
    with st.form("verify"):
        st.write("---")
        st.write("#### 📝 Verification")
        sel_group = st.selectbox("Group", list(BRIDGE_SCHEMA.keys()), index=list(BRIDGE_SCHEMA.keys()).index(item['group']))
        avail_comps = list(BRIDGE_SCHEMA[sel_group].keys())
        sel_comp = st.selectbox("Component", avail_comps, index=avail_comps.index(item['comp']) if item['comp'] in avail_comps else 0)
        avail_defects = BRIDGE_SCHEMA[sel_group][sel_comp]["defects"] + ["No Defect"]
        sel_defect = st.selectbox("Defect", avail_defects, index=avail_defects.index(item['type']) if item['type'] in avail_defects else 0)
        
        v_depth = st.number_input("Confirmed Severity", value=float(ai_depth), format="%.4f")
        note = st.text_area("Note")
        
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id": item['id'], "group": sel_group, "comp": sel_comp, "type": sel_defect,
                "severity": v_depth, "doh": doh, "priority": cv*w, "action": action, "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.idx += 1
            st.rerun()