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
        elif measured_val > 0.10: doh_rating = 2     # > 10% Loss
        elif measured_val > 0.01: doh_rating = 3     # Pitting
        elif measured_val > 0: doh_rating = 4        # Surface Rust
    elif any(x in defect_type for x in ["Scour", "Settlement", "Tilt"]):
        if measured_val > 0.50: doh_rating = 1       
        elif measured_val > 0.20: doh_rating = 2     
        elif measured_val > 0.05: doh_rating = 3     
        elif measured_val > 0: doh_rating = 4

    if defect_type == "No Defect": doh_rating = 5

    mapping_table = {5:1, 4:2, 3:3, 2:4, 1:5, 0:5}
    cv_score = mapping_table.get(doh_rating, 1)

    weight = 1.0
    primary_comps = ["Girder", "Pier", "Cap Beam", "Footing", "Bearing"]
    if any(p in component_name for p in primary_comps):
        weight = 1.5
    
    priority_score = cv_score * weight
    
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
        for by in [2.0, 4.0, 6.0]: add_dense_block([sx-0.2, sx+0.2], [by-0.2, by+0.2], [-1.5, -1.4], density=800)
        for py in [2.5, 5.5]: 
            add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_pier_bot, z_cap_bot], density=500)
            add_dense_block([sx-1.0, sx+1.0], [py-1.0, py+1.0], [z_foot_bot, z_pier_bot], density=300)

    full = np.concatenate(points_list, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    Z += np.random.normal(0, 0.005, size=Z.shape)
    S = np.zeros_like(Z)
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
            elif "Corrosion" in defect_type: severity_factor = 0.15
            
            Z[mask] -= severity_factor
            S[mask] = 1.0 # High severity for viz
            ai_depth = severity_factor

    return X, Y, Z, S, ai_depth

# --- 5. DATA GENERATOR ---
def generate_mock_batch():
    batch = []
    for _ in range(5):
        group = random.choice(list(BRIDGE_SCHEMA.keys()))
        comp = random.choice(list(BRIDGE_SCHEMA[group].keys()))
        defect = random.choice(BRIDGE_SCHEMA[group][comp]["defects"])
        depth = 0.0
        if "Cracking" in defect: depth = random.uniform(0.001, 0.008)
        elif "Spalling" in defect: depth = random.uniform(0.02, 0.15)
        elif "Scour" in defect: depth = random.uniform(0.1, 0.6)
        elif "Corrosion" in defect: depth = random.uniform(0.05, 0.40)
        else: depth = random.uniform(0.0, 0.05)
        
        batch.append({
            "id": f"INS-{random.randint(100,999)}",
            "group": group, "comp": comp,
            "comp_th": BRIDGE_SCHEMA[group][comp]["name_th"],
            "type": defect, "depth": depth
        })
    return batch

# --- 6. DATA HANDLER ---
def get_inspection_data(uploaded_file, mock_item):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower() for c in df.columns]
            if {'x','y','z'}.issubset(df.columns):
                if len(df) > 25000: df = df.sample(25000)
                X, Y, Z = df['x'].values, df['y'].values, df['z'].values
                ai_d = abs(np.min(Z) - np.mean(Z)) if len(Z)>0 else 0
                S = np.zeros_like(Z)
                return X, Y, Z, S, ai_d, "Real File Uploaded"
        except: pass
    
    X, Y, Z, S, ai_depth = generate_complex_structure(mock_item['type'], mock_item['comp'])
    return X, Y, Z, S, item['depth'], "Mockup Data"

# --- 7. MAIN APP ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []
if 'mock_data' not in st.session_state: st.session_state.mock_data = generate_mock_batch()

st.sidebar.title("🛠️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Point Cloud (.csv)", type=['csv'])
if st.sidebar.button("🔄 Generate New Batch"):
    st.session_state.mock_data = generate_mock_batch()
    st.session_state.idx = 0; st.rerun()

if st.session_state.results:
    st.sidebar.download_button("📥 Backup Data (CSV)", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8-sig'), "backup.csv", "text/csv")

st.title("🌉 Hybrid Bridge Inspector v6.8 (Detailed Standards)")
st.caption("Standards: DOH Detection ➔ Pellegrini Management | Feature: Full Reference Guide")

if st.session_state.idx >= len(st.session_state.mock_data):
    st.success("✅ All items in batch inspected!")
    df_res = pd.DataFrame(st.session_state.results)
    st.dataframe(df_res)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button("📥 Download Final Report", df_res.to_csv(index=False).encode('utf-8-sig'), f"Report_{datetime.now().strftime('%H%M')}.csv", "text/csv", type="primary")
    with col2:
        if st.button("Start New"): st.session_state.idx=0; st.session_state.results=[]; st.rerun()
    st.stop()

item = st.session_state.mock_data[st.session_state.idx]
X, Y, Z, S, ai_depth, source_txt = get_inspection_data(uploaded_file, item)
doh, cv, w, urgency, action, css = calculate_hybrid_assessment(item['type'], ai_depth, item['comp'])

col_viz, col_data = st.columns([1.8, 1])

with col_viz:
    st.subheader(f"📍 {item['comp']} ({item['comp_th']})")
    
    c1, c2 = st.columns([1, 1])
    with c1: st.caption(f"Defect: {item['type']}")
    with c2: view_mode = st.radio("Display Mode:", ["Elevation", "Severity"], horizontal=True)

    if len(X) > 0:
        color_data = Z if view_mode == "Elevation" else S
        c_scale = 'Jet_r' if view_mode == "Elevation" else 'RdYlGn_r'
        c_title = "Elevation (m)" if view_mode == "Elevation" else "Severity Index"

        fig_3d = go.Figure(data=[go.Scatter3d(
            x=X, y=Y, z=Z, mode='markers', 
            marker=dict(size=3, color=color_data, colorscale=c_scale, opacity=0.8, showscale=True, colorbar=dict(title=c_title, thickness=15, x=1.0))
        )])
        
        slice_pos = st.slider("X-Axis Cut", float(np.min(X)), float(np.max(X)), float(np.mean(X)))
        if view_mode == "Elevation":
            py, pz = np.meshgrid(np.linspace(np.min(Y), np.max(Y), 10), np.linspace(np.min(Z), np.max(Z), 10))
            px = np.full_like(py, slice_pos)
            fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
        
        fig_3d.update_layout(template='plotly_white', height=500, scene=dict(aspectmode='data'), margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_3d, use_container_width=True)
        
        mask = np.abs(X - slice_pos) < 0.2
        fig_sec = go.Figure(go.Scatter(x=Y[mask], y=Z[mask], mode='markers', marker=dict(size=5, color=Z[mask], colorscale='Jet_r', opacity=0.8)))
        fig_sec.update_layout(template='plotly_white', height=200, title=f"Cross-Section (Elevation) at X={slice_pos:.1f}m", margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.error("No Data")

with col_data:
    st.markdown("### 📊 Assessment Card")
    st.markdown(f"""
    <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
        <small>Stage 1: Detection (DOH)</small><br>
        Measured: <b>{ai_depth:.4f}</b><br>Rating: <b>{doh}/5</b>
    </div>
    <div class="arrow-box">⬇️</div>
    <div style="border:1px solid #000080; padding:10px; border-radius:5px; background-color:#f0f4ff;">
        <small>Stage 3: Management</small><br>
        CV Score: <b>{cv}</b> (x{w})<br>Priority: <b>{cv*w:.1f}</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""<div style="margin-top:15px; text-align:center;"><span class="{css}">{urgency}</span><br><h4>{action}</h4></div>""", unsafe_allow_html=True)
    
    # --- UPGRADED EXPANDER: DETAILED STANDARDS ---
    with st.expander("📘 Reference Standards (เกณฑ์การประเมินละเอียด)"):
        t1, t2, t3 = st.tabs(["🇹🇭 DOH Detailed", "🔄 Mapping", "🇪🇺 Algo"])
        
        with t1:
            st.markdown("#### 1. คอนกรีต & ผิวทาง")
            st.markdown("""
            | Defect | Severity (Threshold) | Rating |
            | :--- | :--- | :---: |
            | **Cracking** | กว้าง > 5.0 mm | **1** |
            | | กว้าง 2.0 - 5.0 mm | **2** |
            | | กว้าง 0.3 - 2.0 mm | **3** |
            | | กว้าง < 0.3 mm | **4** |
            | **Spalling** | ลึก > 15 cm / เหล็กขาด | **1** |
            | | ลึก > 10 cm / เหล็กสนิม | **2** |
            | | ลึก > 2.5 cm (ถึงเหล็ก) | **3** |
            | | ลึก < 2.5 cm (ผิวหน้า) | **4** |
            """)
            st.markdown("#### 2. เหล็กเสริม & โครงสร้างเหล็ก")
            st.markdown("""
            | Defect | Severity (Threshold) | Rating |
            | :--- | :--- | :---: |
            | **Corrosion** | หน้าตัดหาย > 30% | **1** |
            | | หน้าตัดหาย > 10% | **2** |
            | | สนิมขุม (Pitting) | **3** |
            | | สนิมผิว (Surface Rust) | **4** |
            """)
            st.markdown("#### 3. ฐานราก & ตอม่อ")
            st.markdown("""
            | Defect | Severity (Threshold) | Rating |
            | :--- | :--- | :---: |
            | **Scour** | ฐานรากลอย > 50% / เสี่ยงพัง | **1** |
            | | ฐานรากลอย / เห็นเสาเข็ม | **2** |
            | | กัดเซาะถึง Footing | **3** |
            | **Settlement** | ทรุด > 50 mm (โครงสร้างร้าว) | **1** |
            | | ทรุด > 10 mm (สังเกตเห็นได้) | **3** |
            """)

        with t2:
            st.write("DOH (5=ดี) ➡️ Pellegrini CV (1=ดี)")
            st.markdown("| Rating | 5 | 4 | 3 | 2 | 1 |")
            st.markdown("| :--- | :---: | :---: | :---: | :---: | :---: |")
            st.markdown("| **CV** | **1** | **2** | **3** | **4** | **5** |")
            
        with t3:
            st.latex(r''' Priority = CV \times Weight ''')
            st.write("- **Weight 1.5:** Primary Members")
            st.write("- **Weight 1.0:** Secondary Members")
            st.info("Score ≥ 6.0: 🔴 Repair Immediately")

    with st.form("verify"):
        st.write("---")
        st.write("#### 📝 Verification")
        sel_group = st.selectbox("Group", list(BRIDGE_SCHEMA.keys()), index=list(BRIDGE_SCHEMA.keys()).index(item['group']))
        avail_comps = list(BRIDGE_SCHEMA[sel_group].keys())
        sel_comp = st.selectbox("Component", avail_comps, index=avail_comps.index(item['comp']) if item['comp'] in avail_comps else 0)
        avail_defects = BRIDGE_SCHEMA[sel_group][sel_comp]["defects"] + ["No Defect"]
        sel_defect = st.selectbox("Defect", avail_defects, index=avail_defects.index(item['type']) if item['type'] in avail_defects else 0)
        v_depth = st.number_input("Severity (Value)", value=float(ai_depth), format="%.4f")
        note = st.text_area("Note")
        
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id": item['id'], "group": sel_group, "comp": sel_comp, "type": sel_defect,
                "severity": v_depth, "doh": doh, "priority": cv*w, "action": action, "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.idx += 1
            st.rerun()