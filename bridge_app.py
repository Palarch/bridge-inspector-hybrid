import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
from datetime import datetime

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Hybrid Bridge Inspector v7.1 (Ultimate)", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px; }
    
    .urgency-1 { background-color: #dc3545; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; } /* Critical */
    .urgency-2 { background-color: #fd7e14; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; } /* High */
    .urgency-3 { background-color: #ffc107; color: black; padding: 6px 15px; border-radius: 20px; font-weight: bold; } /* Medium */
    .urgency-4 { background-color: #28a745; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; } /* Low */
    
    .arrow-box { font-size: 24px; text-align: center; margin: 5px 0; color: #6c757d; }
    .math-box { font-family: 'Courier New', monospace; background-color: #e9ecef; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    /* Table Styling */
    th { background-color: #f1f3f5; }
</style>
""", unsafe_allow_html=True)

# --- 2. BRIDGE DATA SCHEMA ---
BRIDGE_SCHEMA = {
    "Superstructure": {
        "Deck": {"name_th": "พื้นสะพาน", "defects": ["Cracking", "Spalling", "Corrosion (Rebar)", "Wear/Abrasion", "Delamination"]},
        "Girder": {"name_th": "คานตามยาว", "defects": ["Flexure Cracks", "Shear Cracks", "Spalling", "Corrosion (Rebar)", "Excessive Deflection"]},
        "Diaphragm": {"name_th": "ค้ำยันคาน", "defects": ["Cracking", "Spalling"]},
        "Wearing Surface": {"name_th": "ผิวทาง", "defects": ["Potholing", "Rutting", "Map Cracking"]},
        "Expansion Joint": {"name_th": "รอยต่อ", "defects": ["Clog", "Leakage", "Component Failure"]}
    },
    "Substructure": {
        "Cap Beam": {"name_th": "คานรัดหัวเสา", "defects": ["Cracking", "Corrosion (Rebar)", "Spalling"]},
        "Pier": {"name_th": "เสาตอม่อ", "defects": ["Settlement/Tilt", "Scour", "Spalling", "Cracking"]},
        "Footing": {"name_th": "ฐานราก", "defects": ["Scour/Exposure", "Settlement"]},
        "Bearing": {"name_th": "แผ่นรองรับคาน", "defects": ["Deformation", "Corrosion", "Slippage"]}
    }
}

# --- 3. ADVANCED HYBRID ENGINE (v7.0 Logic) ---
def calculate_advanced_hybrid(defect_type, measured_val, component_name):
    # STAGE 1: DOH (Physical Severity)
    doh_rating = 5 # Default
    
    if "Crack" in defect_type:
        if measured_val > 0.005: doh_rating = 1      # >5mm
        elif measured_val > 0.002: doh_rating = 2    # >2mm
        elif measured_val > 0.0003: doh_rating = 3   # >0.3mm
        elif measured_val > 0: doh_rating = 4
    elif any(x in defect_type for x in ["Spalling", "Scaling", "Potholing", "Scour"]):
        if measured_val > 0.15: doh_rating = 1       # >15cm
        elif measured_val > 0.10: doh_rating = 2
        elif measured_val > 0.025: doh_rating = 3
        elif measured_val > 0: doh_rating = 4
    elif any(x in defect_type for x in ["Movement", "Settlement", "Tilt"]):
        if measured_val > 0.05: doh_rating = 1
        elif measured_val > 0.025: doh_rating = 2
        elif measured_val > 0.010: doh_rating = 3
        elif measured_val > 0: doh_rating = 4
    elif "Corrosion" in defect_type:
        if measured_val > 0.30: doh_rating = 1
        elif measured_val > 0.10: doh_rating = 2
        elif measured_val > 0.01: doh_rating = 3
        else: doh_rating = 4

    if defect_type == "No Defect": doh_rating = 5

    # STAGE 2: MAPPING (Invert)
    cv_score = 6 - doh_rating
    if doh_rating == 0: cv_score = 5

    # STAGE 3: EVALUATION (Advanced Priority)
    # 3.1 Component Weight
    w_comp = 1.0
    primary_comps = ["Girder", "Pier", "Cap Beam", "Footing", "Bearing"]
    if any(p in component_name for p in primary_comps): w_comp = 1.5
    
    # 3.2 Defect Criticality (The Upgrade)
    w_defect = 1.0
    if "Shear Cracks" in defect_type: w_defect = 1.5      # Critical Structural
    elif "Settlement" in defect_type: w_defect = 1.4      # Stability Issue
    elif "Scour" in defect_type: w_defect = 1.4           # Stability Issue
    elif "Corrosion" in defect_type: w_defect = 1.2       # Material Loss
    
    priority_score = cv_score * w_comp * w_defect
    
    # Classification
    if priority_score >= 9.0:
        urgency = "Urgency 1 (Critical)"
        action = "⛔ ปิด/ซ่อมทันที (Immediate)"
        css = "urgency-1"
    elif priority_score >= 6.0:
        urgency = "Urgency 2 (High)"
        action = "🔴 ซ่อมเร่งด่วน (Urgent)"
        css = "urgency-1"
    elif priority_score >= 3.5:
        urgency = "Urgency 3 (Medium)"
        action = "🟠 ซ่อมตามแผน (Planned)"
        css = "urgency-2"
    elif priority_score >= 1.5:
        urgency = "Urgency 4 (Low)"
        action = "🟢 เฝ้าระวัง (Monitor)"
        css = "urgency-3"
    else:
        urgency = "Normal"
        action = "✅ ปกติ"
        css = "urgency-4"

    return doh_rating, cv_score, w_comp, w_defect, priority_score, urgency, action, css

# --- 4. STRUCTURE GENERATOR (Volumetric & Severity Map) ---
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
        xe = np.linspace(x_lim[0], x_lim[1], 10); ye = np.linspace(y_lim[0], y_lim[1], 10)
        Xg, Yg = np.meshgrid(xe, ye)
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[0]).flatten()], axis=1))
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[1]).flatten()], axis=1))
        points_list.append(np.stack([xx, yy, zz], axis=1))

    L = 12.0; W = 8.0
    z_d_bot = -0.3; z_g_bot = -1.5; z_c_bot = -2.5; z_p_bot = -6.0

    add_dense_block([0, L], [0, W], [z_d_bot, 0], density=600)
    for y in [2.0, 4.0, 6.0]: add_dense_block([0, L], [y-0.3, y+0.3], [z_g_bot, z_d_bot], density=500)
    for sx in [2.0, 10.0]:
        add_dense_block([sx-0.6, sx+0.6], [0.5, W-0.5], [z_c_bot, -1.5], density=600)
        for py in [2.5, 5.5]: add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_p_bot, z_c_bot], density=500)

    full = np.concatenate(points_list, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    Z += np.random.normal(0, 0.005, size=Z.shape)
    S = np.zeros_like(Z)
    ai_depth = 0.0
    
    if defect_type != "No Defect":
        mask = np.zeros_like(Z, dtype=bool)
        if "Deck" in component_name: mask = (Z > -0.1) & ((X-6)**2 + (Y-4)**2 < 2.5)
        elif "Girder" in component_name: mask = (Z < z_g_bot+0.5) & (abs(Y-4.0)<0.4) & (abs(X-6.0)<0.3)
        elif "Cap" in component_name: mask = (abs(X-2.0)<0.7) & (Z>z_c_bot) & (Y<2.0)
        elif "Pier" in component_name: mask = (abs(X-2.0)<0.5) & (Z<z_c_bot-1) & (Y<3.0)
        
        if np.any(mask):
            sf = 0.02
            if "Crack" in defect_type: sf = 0.005
            elif "Spall" in defect_type: sf = 0.15
            elif "Scour" in defect_type: sf = 0.30
            elif "Corros" in defect_type: sf = 0.15
            Z[mask] -= sf
            S[mask] = 1.0 
            ai_depth = sf

    return X, Y, Z, S, ai_depth

# --- 5. DATA GENERATOR ---
def generate_mock_batch():
    batch = []
    scenarios = [
        ("Superstructure", "Girder", "Shear Cracks", 0.006),
        ("Superstructure", "Deck", "Spalling", 0.03),
        ("Substructure", "Pier", "Settlement/Tilt", 0.06),
        ("Substructure", "Cap Beam", "Corrosion (Rebar)", 0.15),
        ("Superstructure", "Wearing Surface", "Potholing", 0.05)
    ]
    for i, (grp, cmp, typ, d) in enumerate(scenarios):
        batch.append({
            "id": f"INS-{100+i}", "group": grp, "comp": cmp,
            "comp_th": BRIDGE_SCHEMA[grp][cmp]["name_th"], "type": typ, "depth": d
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
if st.sidebar.button("🔄 Reset Batch"):
    st.session_state.mock_data = generate_mock_batch()
    st.session_state.idx = 0; st.rerun()

if st.session_state.results:
    st.sidebar.download_button("📥 Backup Data", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8-sig'), "backup.csv", "text/csv")

st.title("🌉 Hybrid Bridge Inspector v7.1 (Ultimate)")
st.caption("Standards: DOH Detection ➔ Pellegrini Management (Advanced Algorithm)")

if st.session_state.idx >= len(st.session_state.mock_data):
    st.success("✅ Inspection Completed!")
    df_res = pd.DataFrame(st.session_state.results)
    st.dataframe(df_res)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button("📥 Final Report", df_res.to_csv(index=False).encode('utf-8-sig'), f"Report_{datetime.now().strftime('%H%M')}.csv", "text/csv", type="primary")
    with col2:
        if st.button("Start New"): st.session_state.idx=0; st.session_state.results=[]; st.rerun()
    st.stop()

item = st.session_state.mock_data[st.session_state.idx]
X, Y, Z, S, ai_depth, source_txt = get_inspection_data(uploaded_file, item)
doh, cv, w_comp, w_defect, priority, urgency, action, css = calculate_advanced_hybrid(item['type'], ai_depth, item['comp'])

col_viz, col_data = st.columns([1.8, 1])

with col_viz:
    st.subheader(f"📍 {item['comp']} ({item['comp_th']})")
    
    c1, c2 = st.columns([1, 1])
    with c1: st.caption(f"Defect: {item['type']}")
    with c2: view_mode = st.radio("View Mode:", ["Elevation", "Severity Map"], horizontal=True)

    if len(X) > 0:
        c_data = Z if view_mode == "Elevation" else S
        c_scale = 'Jet_r' if view_mode == "Elevation" else 'RdYlGn_r'
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=X, y=Y, z=Z, mode='markers', 
            marker=dict(size=3, color=c_data, colorscale=c_scale, opacity=0.8, showscale=True, colorbar=dict(thickness=15, x=1.0))
        )])
        
        slice_pos = st.slider("Section Cut (X)", float(np.min(X)), float(np.max(X)), float(np.mean(X)))
        if view_mode == "Elevation":
            py, pz = np.meshgrid(np.linspace(np.min(Y), np.max(Y), 10), np.linspace(np.min(Z), np.max(Z), 10))
            px = np.full_like(py, slice_pos)
            fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
        
        fig_3d.update_layout(template='plotly_white', height=500, scene=dict(aspectmode='data'), margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_3d, use_container_width=True)
        
        mask = np.abs(X - slice_pos) < 0.2
        fig_sec = go.Figure(go.Scatter(x=Y[mask], y=Z[mask], mode='markers', marker=dict(size=5, color=Z[mask], colorscale='Jet_r', opacity=0.8)))
        fig_sec.update_layout(template='plotly_white', height=200, title=f"Section at X={slice_pos:.1f}m", margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_sec, use_container_width=True)
    else: st.error("No Data")

with col_data:
    st.markdown("### 📊 Assessment Card")
    st.markdown(f"""
    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px;">
        <small>1. DOH Detection</small><br>
        Measured: <b>{ai_depth:.4f}</b> ➔ Rating: <b>{doh}/5</b>
    </div>
    <div style="border:1px solid #000080; padding:10px; border-radius:5px; background-color:#f0f4ff;">
        <small>2. Advanced Evaluation</small><br>
        CV Score: <b>{cv}</b> (Inverted)<br>
        <span class="math-box">x {w_comp}</span> (Component)<br>
        <span class="math-box">x {w_defect}</span> (Defect Risk)<br>
        <hr style="margin:5px 0">
        <b>Priority Index: {priority:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""<div style="margin-top:15px; text-align:center;"><span class="{css}">{urgency}</span><br><h4>{action}</h4></div>""", unsafe_allow_html=True)
    
    # --- DETAILED REFERENCE GUIDE (Restored & Updated) ---
    with st.expander("📘 Reference Standards (คู่มือเกณฑ์การประเมินละเอียด)"):
        t1, t2, t3 = st.tabs(["🇹🇭 DOH Detailed", "🔄 Mapping", "🇪🇺 Advanced Algo"])
        
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
            st.latex(r''' Priority = CV \times W_{comp} \times W_{defect} ''')
            st.markdown("**1. Component Weight ($W_{comp}$):**")
            st.write("- **1.5:** Primary Members (Girder, Pier, Cap)")
            st.write("- **1.0:** Secondary Members")
            st.markdown("**2. Defect Criticality ($W_{defect}$):**")
            st.write("- **1.5:** Shear Cracks (อันตรายสูงสุด)")
            st.write("- **1.4:** Settlement / Scour (เสี่ยงถล่ม)")
            st.write("- **1.2:** Corrosion")
            st.write("- **1.0:** General (Spalling, Wear)")

    with st.form("verify"):
        st.write("---")
        sel_group = st.selectbox("Group", list(BRIDGE_SCHEMA.keys()), index=list(BRIDGE_SCHEMA.keys()).index(item['group']))
        avail_comps = list(BRIDGE_SCHEMA[sel_group].keys())
        sel_comp = st.selectbox("Component", avail_comps, index=avail_comps.index(item['comp']) if item['comp'] in avail_comps else 0)
        avail_defects = BRIDGE_SCHEMA[sel_group][sel_comp]["defects"] + ["No Defect"]
        sel_defect = st.selectbox("Defect", avail_defects, index=avail_defects.index(item['type']) if item['type'] in avail_defects else 0)
        v_depth = st.number_input("Severity (Value)", value=float(ai_depth), format="%.4f")
        note = st.text_area("Note")
        
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id": item['id'], "comp": sel_comp, "type": sel_defect,
                "severity": v_depth, "doh": doh, "priority": priority, "action": action, "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.idx += 1
            st.rerun()