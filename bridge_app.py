import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
from datetime import datetime

# --- 1. CONFIG & CSS SETUP ---
st.set_page_config(page_title="Hybrid Bridge Inspector v6.5 (Master)", layout="wide")

st.markdown("""
<style>
    /* Clean Light Theme */
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px; }
    
    /* Urgency Badges */
    .urgency-1 { background-color: #dc3545; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    .urgency-2 { background-color: #fd7e14; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    .urgency-3 { background-color: #28a745; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    
    .arrow-box { font-size: 24px; text-align: center; margin: 5px 0; color: #6c757d; }
    
    /* Table Styling inside Expander */
    thead tr th:first-child { display:none }
    tbody th { display:none }
</style>
""", unsafe_allow_html=True)

# --- 2. BRIDGE DATA SCHEMA (DATABASE) ---
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

# --- 3. HYBRID LOGIC ENGINE (Aligned with DOH Standards) ---
def calculate_hybrid_assessment(defect_type, measured_val, component_name):
    # STAGE 1: DOH DETECTION (5-0 Scale)
    doh_rating = 5 # Default Good
    
    # Logic ตามเกณฑ์ใน Expander
    if "Crack" in defect_type:
        if measured_val > 0.005: doh_rating = 1      # > 5 mm
        elif measured_val > 0.002: doh_rating = 2    # > 2 mm
        elif measured_val > 0.0003: doh_rating = 3   # > 0.3 mm
        elif measured_val > 0: doh_rating = 4        # < 0.3 mm
    elif any(x in defect_type for x in ["Spalling", "Scaling", "Honeycomb", "Potholing", "Scour"]):
        if measured_val > 0.15: doh_rating = 1       # > 15 cm
        elif measured_val > 0.10: doh_rating = 2     # > 10 cm
        elif measured_val > 0.025: doh_rating = 3    # > 2.5 cm
        elif measured_val > 0: doh_rating = 4
    elif any(x in defect_type for x in ["Movement", "Settlement", "Deflection", "Tilt"]):
        if measured_val > 0.05: doh_rating = 1       # > 50 mm
        elif measured_val > 0.01: doh_rating = 3     # > 10 mm
        elif measured_val > 0: doh_rating = 4
    else: 
        if measured_val > 0.1: doh_rating = 2
        elif measured_val > 0: doh_rating = 3

    if defect_type == "No Defect": doh_rating = 5

    # STAGE 2: MAPPING (Invert Scale)
    mapping_table = {5:1, 4:2, 3:3, 2:4, 1:5, 0:5}
    cv_score = mapping_table.get(doh_rating, 1)

    # STAGE 3: EVALUATION (Pellegrini Weighting)
    weight = 1.0
    primary_comps = ["Girder", "Pier & Pier Wall", "Cap Beam", "Footing", "Bearing"]
    if any(p in component_name for p in primary_comps):
        weight = 1.5
    
    priority_score = cv_score * weight
    
    # Determine Urgency
    if priority_score >= 6.0:
        return doh_rating, cv_score, weight, "Urgency 1 (High)", "ซ่อมทันที (Repair Immediately)", "urgency-1"
    elif priority_score >= 3.0:
        return doh_rating, cv_score, weight, "Urgency 2 (Medium)", "ซ่อมระยะสั้น (Short-term Repair)", "urgency-2"
    else:
        return doh_rating, cv_score, weight, "Urgency 3 (Low)", "เฝ้าระวัง (Monitor)", "urgency-3"

# --- 4. STRUCTURE GENERATOR (Volumetric & Optimized) ---
def generate_complex_structure(defect_type, component_name):
    points_list = []
    
    # Function to create dense volume blocks
    def add_dense_block(x_lim, y_lim, z_lim, density=400): 
        vol = (x_lim[1]-x_lim[0]) * (y_lim[1]-y_lim[0]) * (z_lim[1]-z_lim[0])
        n_points = int(density * vol)
        
        # Safety Limit to prevent WebGL Crash
        if n_points > 4500: n_points = 4500
        if n_points < 200: n_points = 200
        
        xx = np.random.uniform(x_lim[0], x_lim[1], n_points)
        yy = np.random.uniform(y_lim[0], y_lim[1], n_points)
        zz = np.random.uniform(z_lim[0], z_lim[1], n_points)
        
        # Add wireframe-like edges
        xe = np.linspace(x_lim[0], x_lim[1], 10)
        ye = np.linspace(y_lim[0], y_lim[1], 10)
        Xg, Yg = np.meshgrid(xe, ye)
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[0]).flatten()], axis=1))
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[1]).flatten()], axis=1))
        points_list.append(np.stack([xx, yy, zz], axis=1))

    L = 12.0; W = 8.0
    z_deck_bot = -0.3; z_girder_bot = -1.5; z_cap_bot = -2.5; z_pier_bot = -6.0; z_foot_bot = -7.0

    # Build Geometry
    add_dense_block([0, L], [0, W], [z_deck_bot, 0], density=600) # Deck
    for y in [2.0, 4.0, 6.0]:
        add_dense_block([0, L], [y-0.3, y+0.3], [z_girder_bot, z_deck_bot], density=500) # Girder
    
    for sx in [2.0, 10.0]:
        add_dense_block([sx-0.6, sx+0.6], [0.5, W-0.5], [z_cap_bot, -1.5], density=600) # Cap Beam
        for by in [2.0, 4.0, 6.0]: 
             add_dense_block([sx-0.2, sx+0.2], [by-0.2, by+0.2], [-1.5, -1.4], density=800) # Bearing
        for py in [2.5, 5.5]: 
            add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_pier_bot, z_cap_bot], density=500) # Pier
            add_dense_block([sx-1.0, sx+1.0], [py-1.0, py+1.0], [z_foot_bot, z_pier_bot], density=300) # Footing

    full = np.concatenate(points_list, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    # Simulate Defect Visuals
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
            if "Cracking" in defect_type: severity_factor = 0.005
            elif "Spalling" in defect_type: severity_factor = 0.15
            elif "Scour" in defect_type: severity_factor = 0.30
            
            Z[mask] -= severity_factor
            ai_depth = severity_factor

    return X, Y, Z, ai_depth

# --- 5. DATA GENERATOR (Mock Batch) ---
def generate_mock_batch():
    batch = []
    for _ in range(5):
        group = random.choice(list(BRIDGE_SCHEMA.keys()))
        comp = random.choice(list(BRIDGE_SCHEMA[group].keys()))
        defect = random.choice(BRIDGE_SCHEMA[group][comp]["defects"])
        depth = 0.0
        if "Cracking" in defect: depth = random.uniform(0.001, 0.008)
        elif "Spalling" in defect: depth = random.uniform(0.02, 0.15)
        elif "Scour" in defect: depth = random.uniform(0.1, 0.5)
        else: depth = random.uniform(0.0, 0.05)
        
        batch.append({
            "id": f"INS-{random.randint(100,999)}",
            "group": group, "comp": comp,
            "comp_th": BRIDGE_SCHEMA[group][comp]["name_th"],
            "type": defect, "depth": depth
        })
    return batch

# --- 6. DATA HANDLER (File vs Mock) ---
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
    
    X, Y, Z, ai_depth = generate_complex_structure(mock_item['type'], mock_item['comp'])
    return X, Y, Z, item['depth'], "Mockup Data"

# --- 7. MAIN APP ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []
if 'mock_data' not in st.session_state: st.session_state.mock_data = generate_mock_batch()

# SIDEBAR
st.sidebar.title("🛠️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Point Cloud (.csv)", type=['csv'])
if st.sidebar.button("🔄 Generate New Batch"):
    st.session_state.mock_data = generate_mock_batch()
    st.session_state.idx = 0
    st.rerun()

# Sidebar Backup
if st.session_state.results:
    st.sidebar.download_button("📥 Backup Data (CSV)", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8-sig'), "backup.csv", "text/csv")

# MAIN CONTENT
st.title("🌉 Hybrid Bridge Inspector v6.5")
st.caption("Standards: DOH Detection ➔ Pellegrini Management | Feature: Full Schema + Real File Support")

# End Screen
if st.session_state.idx >= len(st.session_state.mock_data):
    st.success("✅ All items in batch inspected!")
    df_res = pd.DataFrame(st.session_state.results)
    st.dataframe(df_res)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="📥 Download Final Report (CSV)",
            data=df_res.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"Bridge_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )
    with col2:
        if st.button("Start New Inspection"): 
            st.session_state.idx=0
            st.session_state.results=[]
            st.rerun()
    st.stop()

# Processing
item = st.session_state.mock_data[st.session_state.idx]
X, Y, Z, ai_depth, source_txt = get_inspection_data(uploaded_file, item)
doh, cv, w, urgency, action, css = calculate_hybrid_assessment(item['type'], ai_depth, item['comp'])

# Layout
col_viz, col_data = st.columns([1.8, 1])

with col_viz:
    st.subheader(f"📍 {item['comp']} ({item['comp_th']})")
    st.caption(f"Defect: {item['type']} | Source: {source_txt}")
    
    # 1. SLIDER & 2D (Fixed)
    st.markdown("##### ✂️ Cross-Section Analyzer")
    if len(X) > 0:
        slice_pos = st.slider("X-Axis Cut", float(np.min(X)), float(np.max(X)), float(np.mean(X)))
        mask = np.abs(X - slice_pos) < 0.2
        
        fig_sec = go.Figure(go.Scatter(x=Y[mask], y=Z[mask], mode='markers', marker=dict(size=5, color=Z[mask], colorscale='Jet_r', opacity=0.8)))
        fig_sec.update_layout(template='plotly_white', height=250, title=f"Section at X={slice_pos:.1f}m", margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_sec, use_container_width=True)

        # 2. 3D GRAPH
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=X, y=Y, z=Z, 
            mode='markers', 
            marker=dict(size=3, color=Z, colorscale='Jet_r', opacity=0.7, showscale=True, colorbar=dict(title="Elevation (m)", thickness=15, x=1.0))
        )])
        
        py, pz = np.meshgrid(np.linspace(np.min(Y), np.max(Y), 10), np.linspace(np.min(Z), np.max(Z), 10))
        px = np.full_like(py, slice_pos)
        fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
        
        fig_3d.update_layout(template='plotly_white', height=500, scene=dict(aspectmode='data'), margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.error("No Data Points Found")

with col_data:
    st.markdown("### 📊 Assessment Card")
    
    st.markdown(f"""
    <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
        <small>Stage 1: Detection (DOH)</small><br>
        Measured Severity: <b>{ai_depth:.4f} m</b><br>
        Rating: <b>{doh} / 5</b>
    </div>
    <div class="arrow-box">⬇️</div>
    <div style="border:1px solid #000080; padding:10px; border-radius:5px; background-color:#f0f4ff;">
        <small>Stage 3: Management</small><br>
        CV Score: <b>{cv}</b> (Weight x{w})<br>
        Priority Score: <b>{cv*w:.1f}</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""<div style="margin-top:15px; text-align:center;"><span class="{css}">{urgency}</span><br><h4>{action}</h4></div>""", unsafe_allow_html=True)
    
    # --- UPGRADED EXPANDER: Detailed Reference ---
    with st.expander("📘 Reference Standards (คู่มือเกณฑ์การประเมิน)"):
        t1, t2, t3 = st.tabs(["🇹🇭 DOH Criteria", "🔄 Mapping Logic", "🇪🇺 Pellegrini Algo"])
        
        with t1:
            st.markdown("#### เกณฑ์กรมทางหลวง (DOH 5-0)")
            st.markdown("""
            | Defect | Severity (Threshold) | Rating |
            | :--- | :--- | :---: |
            | **1. รอยร้าว (Cracking)** | กว้าง < 0.3 mm | **4 (Fair)** |
            | | กว้าง 0.3 - 2.0 mm | **3 (Poor)** |
            | | กว้าง > 2.0 mm | **2 (Serious)** |
            | | กว้าง > 5.0 mm | **1 (Critical)** |
            |---|---|---|
            | **2. การหลุดล่อน (Spalling)** | ลึก < 25 mm | **4 (Fair)** |
            | | ลึก 25 - 50 mm | **3 (Poor)** |
            | | พื้นที่กว้าง / เหล็กสนิม | **2 (Serious)** |
            | | คอนกรีตกระเทาะหลุด | **1 (Critical)** |
            """)
            st.caption("*Rating 5 = Good/New")

        with t2:
            st.markdown("#### ตารางแปลงค่า (Invert Scale)")
            st.write("DOH (5=ดี) ➡️ Pellegrini CV (1=ดี)")
            st.markdown("| DOH Rating | 5 | 4 | 3 | 2 | 1 | 0 |")
            st.markdown("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            st.markdown("| **CV Score** | **1** | **2** | **3** | **4** | **5** | **5** |")
            
        with t3:
            st.markdown("#### สมการบริหารจัดการ")
            st.latex(r''' Priority = CV \times Weight ''')
            st.write("- **Weight 1.5:** Primary Members (Girder, Pier)")
            st.write("- **Weight 1.0:** Secondary Members (Deck, Railing)")
            st.markdown("**เกณฑ์ตัดสินใจ:**")
            st.markdown("- **> 6.0:** 🔴 Repair Immediately")
            st.markdown("- **3.0 - 5.5:** 🟠 Short-term Repair")
            st.markdown("- **< 3.0:** 🟢 Monitor")

    with st.form("verify"):
        st.write("---")
        st.write("#### 📝 Verification")
        
        sel_group = st.selectbox("Group", list(BRIDGE_SCHEMA.keys()), index=list(BRIDGE_SCHEMA.keys()).index(item['group']))
        avail_comps = list(BRIDGE_SCHEMA[sel_group].keys())
        sel_comp = st.selectbox("Component", avail_comps, index=avail_comps.index(item['comp']) if item['comp'] in avail_comps else 0)
        avail_defects = BRIDGE_SCHEMA[sel_group][sel_comp]["defects"] + ["No Defect"]
        sel_defect = st.selectbox("Defect", avail_defects, index=avail_defects.index(item['type']) if item['type'] in avail_defects else 0)
        
        v_depth = st.number_input("Confirmed Severity (m)", value=float(ai_depth), format="%.4f")
        note = st.text_area("Note")
        
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id": item['id'], "group": sel_group, "comp": sel_comp,
                "type": sel_defect, "severity": v_depth,
                "doh": doh, "priority": cv*w, "action": action, "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.idx += 1
            st.rerun()