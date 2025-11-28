import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Hybrid Bridge Inspector v9.3 (Detailed)", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px; }
    
    .urgency-1 { background-color: #dc3545; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    .urgency-2 { background-color: #fd7e14; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    .urgency-3 { background-color: #28a745; color: white; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 16px; }
    
    .arrow-box { font-size: 24px; text-align: center; margin: 5px 0; color: #6c757d; }
    .math-box { font-family: 'Courier New', monospace; background-color: #e9ecef; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    th { background-color: #f1f3f5; }
</style>
""", unsafe_allow_html=True)

# --- 2. BRIDGE DATA SCHEMA ---
BRIDGE_SCHEMA = {
    "Superstructure": {
        "Deck": {"name_th": "พื้นสะพาน", "defects": ["Cracking", "Scaling", "Delamination", "Spalling", "Efflorescence", "Honeycomb", "Reinforcement Corrosion"]},
        "Girder": {"name_th": "คานตามยาว", "defects": ["Flexure Cracks", "Shear Cracks", "Excessive Deflection", "Spalling", "Collision Damage", "Reinforcement Corrosion"]},
        "Diaphragm": {"name_th": "ค้ำยันคาน", "defects": ["Cracking", "Abrasion", "Delamination"]},
        "Wearing Surface": {"name_th": "ผิวทาง", "defects": ["Potholing", "Rutting", "Map Cracking"]},
        "Expansion Joint": {"name_th": "รอยต่อ", "defects": ["Clog", "Leakage", "Component Failure"]}
    },
    "Substructure": {
        "Cap Beam": {"name_th": "คานรัดหัวเสา", "defects": ["Cracking", "Corrosion (Rebar)", "Spalling", "Honeycomb"]},
        "Pier": {"name_th": "เสาตอม่อ", "defects": ["Settlement/Tilt", "Scour", "Spalling", "Cracking"]},
        "Footing": {"name_th": "ฐานราก", "defects": ["Scour/Exposure", "Settlement"]},
        "Bearing": {"name_th": "แผ่นรองรับคาน", "defects": ["Deformation", "Corrosion", "Slippage"]}
    }
}

# --- 3. ADVANCED HYBRID LOGIC ---
def calculate_advanced_hybrid(defect_type, measured_val, component_name):
    # STAGE 1: DOH (Physical Severity)
    doh_rating = 5 
    if "Crack" in defect_type:
        if measured_val > 0.005: doh_rating = 1
        elif measured_val > 0.002: doh_rating = 2
        elif measured_val > 0.0003: doh_rating = 3
        elif measured_val > 0: doh_rating = 4
    elif any(x in defect_type for x in ["Spalling", "Scaling", "Potholing", "Scour"]):
        if measured_val > 0.15: doh_rating = 1
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

    # STAGE 2: MAPPING
    cv_score = 6 - doh_rating
    if doh_rating == 0: cv_score = 5

    # STAGE 3: EVALUATION
    w_comp = 1.0
    if any(p in component_name for p in ["Girder", "Pier", "Cap Beam", "Footing", "Bearing"]): w_comp = 1.5
    
    w_defect = 1.0
    if "Shear" in defect_type: w_defect = 1.5
    elif "Settlement" in defect_type or "Scour" in defect_type: w_defect = 1.4
    elif "Corrosion" in defect_type: w_defect = 1.2
    
    priority_score = cv_score * w_comp * w_defect
    
    if priority_score >= 9.0:
        return doh_rating, cv_score, w_comp, w_defect, priority_score, "Urgency 1 (Critical)", "⛔ ปิด/ซ่อมทันที", "urgency-1"
    elif priority_score >= 6.0:
        return doh_rating, cv_score, w_comp, w_defect, priority_score, "Urgency 2 (High)", "🔴 ซ่อมเร่งด่วน", "urgency-1"
    elif priority_score >= 3.5:
        return doh_rating, cv_score, w_comp, w_defect, priority_score, "Urgency 3 (Medium)", "🟠 ซ่อมตามแผน", "urgency-2"
    elif priority_score >= 1.5:
        return doh_rating, cv_score, w_comp, w_defect, priority_score, "Urgency 4 (Low)", "🟢 เฝ้าระวัง", "urgency-3"
    else:
        return doh_rating, cv_score, w_comp, w_defect, priority_score, "Normal", "✅ ปกติ", "urgency-4"

# --- 4. STRUCTURE GENERATOR ---
@st.cache_data(show_spinner=False)
def generate_complex_structure(defect_type, component_name, _seed=0):
    points_list = []
    def add_dense_block(x_lim, y_lim, z_lim, density=400): 
        vol = (x_lim[1]-x_lim[0]) * (y_lim[1]-y_lim[0]) * (z_lim[1]-z_lim[0])
        n_points = int(density * vol)
        if n_points > 4500: n_points = 4500
        if n_points < 200: n_points = 200
        xx = np.random.uniform(x_lim[0], x_lim[1], n_points)
        yy = np.random.uniform(y_lim[0], y_lim[1], n_points)
        zz = np.random.uniform(z_lim[0], z_lim[1], n_points)
        xe = np.linspace(x_lim[0], x_lim[1], 10); ye = np.linspace(y_lim[0], y_lim[1], 10)
        Xg, Yg = np.meshgrid(xe, ye)
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[0]).flatten()], axis=1))
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[1]).flatten()], axis=1))
        points_list.append(np.stack([xx, yy, zz], axis=1))

    L=12.0; W=8.0; z_d_bot=-0.3; z_g_bot=-1.5; z_c_bot=-2.5; z_p_bot=-6.0
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
            sf = 0.02; 
            if "Crack" in defect_type: sf = 0.005
            elif "Spall" in defect_type: sf = 0.15
            elif "Scour" in defect_type: sf = 0.30
            elif "Corros" in defect_type: sf = 0.15
            Z[mask] -= sf; S[mask] = 1.0; ai_depth = sf

    return X, Y, Z, S, ai_depth

# --- 5. HANDLER & MOCK ---
def get_inspection_data(input_method, file_input, mock_item):
    if input_method == "Local Path" and file_input:
        clean_path = file_input.strip().strip('"').strip("'")
        if os.path.exists(clean_path):
            try:
                df = pd.read_csv(clean_path, nrows=50000)
                df.columns = [c.lower() for c in df.columns]
                if {'x','y','z'}.issubset(df.columns):
                    X, Y, Z = df['x'].values, df['y'].values, df['z'].values
                    ai_d = abs(np.min(Z) - np.mean(Z)) if len(Z)>0 else 0
                    S = np.zeros_like(Z)
                    return X, Y, Z, S, ai_d, f"Local: {os.path.basename(clean_path)}"
            except: pass
    elif input_method == "Browser Upload" and file_input:
        try:
            df = pd.read_csv(file_input, nrows=50000)
            df.columns = [c.lower() for c in df.columns]
            if {'x','y','z'}.issubset(df.columns):
                X, Y, Z = df['x'].values, df['y'].values, df['z'].values
                ai_d = abs(np.min(Z) - np.mean(Z)) if len(Z)>0 else 0
                S = np.zeros_like(Z)
                return X, Y, Z, S, ai_d, "Uploaded File"
        except: pass
    
    X, Y, Z, S, ai_depth = generate_complex_structure(mock_item['type'], mock_item['comp'], _seed=mock_item['id'])
    return X, Y, Z, S, item['depth'], "Mockup Data"

def generate_mock_batch():
    batch = []
    scenarios = [("Superstructure","Girder","Shear Cracks",0.006), ("Superstructure","Deck","Spalling",0.03), ("Substructure","Pier","Settlement/Tilt",0.06), ("Substructure","Cap Beam","Corrosion",0.15)]
    for i, (grp, cmp, typ, d) in enumerate(scenarios):
        batch.append({"id": f"INS-{100+i}", "group": grp, "comp": cmp, "comp_th": BRIDGE_SCHEMA[grp][cmp]["name_th"], "type": typ, "depth": d})
    return batch

# --- 6. MAIN APP ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []
if 'mock_data' not in st.session_state: st.session_state.mock_data = generate_mock_batch()
if 'manual_depth' not in st.session_state: st.session_state.manual_depth = None

st.sidebar.title("🛠️ Control Panel")
input_method = st.sidebar.radio("Source:", ["Browser Upload", "Local Path"])
file_input = st.sidebar.file_uploader("CSV", type=['csv']) if input_method == "Browser Upload" else st.sidebar.text_input("File Path")
if st.sidebar.button("🔄 Reset"): st.session_state.mock_data = generate_mock_batch(); st.session_state.idx = 0; st.session_state.results = []; st.rerun()
if st.session_state.results: st.sidebar.download_button("📥 Backup CSV", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8-sig'), "backup.csv", "text/csv")

st.title("🌉 Hybrid Bridge Inspector v9.3 (Detailed Ref)")
if st.session_state.idx >= len(st.session_state.mock_data):
    st.success("✅ Completed!"); st.dataframe(pd.DataFrame(st.session_state.results))
    col1, col2 = st.columns([1, 1])
    with col1: st.download_button("📥 Final Report", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8-sig'), f"Report.csv", "text/csv", type="primary")
    with col2: 
        if st.button("Start New"): st.session_state.idx=0; st.session_state.results=[]; st.rerun()
    st.stop()

item = st.session_state.mock_data[st.session_state.idx]
X, Y, Z, S, ai_depth, source_txt = get_inspection_data(input_method, file_input, item)
curr_d = st.session_state.manual_depth if st.session_state.manual_depth is not None else ai_depth
doh, cv, w_c, w_d, prio, urg, act, css = calculate_advanced_hybrid(item['type'], curr_d, item['comp'])

col_viz, col_data = st.columns([1.8, 1])

with col_viz:
    st.subheader(f"📍 {item['comp']} ({item['comp_th']})")
    
    c1, c2 = st.columns([1, 1])
    with c1: st.caption(f"Defect: {item['type']}")
    with c2: view_mode = st.radio("View:", ["Elevation", "Severity Map"], horizontal=True)
    
    # 3-Axis Selector
    slice_axis = st.selectbox("Slicing Axis:", ["X-Axis (Cross)", "Y-Axis (Longitudinal)", "Z-Axis (Plan)"])

    if len(X) > 0:
        c_data = Z if view_mode == "Elevation" else S
        c_scale = 'Jet_r' if view_mode == "Elevation" else 'RdYlGn_r'
        
        # Axis Logic
        if slice_axis.startswith("X"):
            sl_lbl, mask_col = "X-Cut", X
            h_data, v_data, h_lbl, v_lbl = Y, Z, "Width (Y)", "Height (Z)"
        elif slice_axis.startswith("Y"):
            sl_lbl, mask_col = "Y-Cut", Y
            h_data, v_data, h_lbl, v_lbl = X, Z, "Length (X)", "Height (Z)"
        else:
            sl_lbl, mask_col = "Z-Cut", Z
            h_data, v_data, h_lbl, v_lbl = X, Y, "Length (X)", "Width (Y)"

        # 3D
        fig_3d = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=3, color=c_data, colorscale=c_scale, opacity=0.8, showscale=True, colorbar=dict(thickness=15, x=1.0)))])
        slice_pos = st.slider(sl_lbl, float(np.min(mask_col)), float(np.max(mask_col)), float(np.mean(mask_col)), step=0.1, key="slice_slider")
        
        # Plane
        if view_mode == "Elevation":
            if slice_axis.startswith("X"): py, pz = np.meshgrid(np.linspace(np.min(Y), np.max(Y), 10), np.linspace(np.min(Z), np.max(Z), 10)); px = np.full_like(py, slice_pos); fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
            elif slice_axis.startswith("Y"): px, pz = np.meshgrid(np.linspace(np.min(X), np.max(X), 10), np.linspace(np.min(Z), np.max(Z), 10)); py = np.full_like(px, slice_pos); fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
            else: px, py = np.meshgrid(np.linspace(np.min(X), np.max(X), 10), np.linspace(np.min(Y), np.max(Y), 10)); pz = np.full_like(px, slice_pos); fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
        
        fig_3d.update_layout(template='plotly_white', height=400, scene=dict(aspectmode='data'), margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 2D & Measure
        st.markdown(f"##### 📐 2D Section: {slice_axis}")
        mask = np.abs(mask_col - slice_pos) < 0.8
        
        with st.expander("🛠️ Measurement Tool", expanded=False):
            m1, m2 = st.columns(2)
            h_vals, v_vals = h_data[mask], v_data[mask]
            if len(h_vals) > 0:
                h0, h1 = float(np.min(h_vals)), float(np.max(h_vals))
                v0, v1 = float(np.min(v_vals)), float(np.max(v_vals))
                with m1:
                    hm = st.slider(f"Measure {h_lbl}", h0, h1, (h0, h1), step=0.01)
                    mw = hm[1] - hm[0]
                with m2:
                    vm = st.slider(f"Measure {v_lbl}", v0, v1, (v0, v1), step=0.01)
                    mh = vm[1] - vm[0]
                
                st.info(f"📏 Size: **{mw:.3f}** x **{mh:.3f} m**")
                if st.button(f"Use {v_lbl}"): st.session_state.manual_depth = mh; st.rerun()
                if st.button(f"Use {h_lbl}"): st.session_state.manual_depth = mw; st.rerun()
                
                # 2D Plot
                fig_sec = go.Figure()
                fig_sec.add_trace(go.Scatter(x=h_data[mask], y=v_data[mask], mode='markers', marker=dict(size=5, color=Z[mask] if view_mode=="Elevation" else S[mask], colorscale=c_scale)))
                fig_sec.add_shape(type="rect", x0=hm[0], y0=vm[0], x1=hm[1], y1=vm[1], line=dict(color="Red", width=2, dash="dash"))
                fig_sec.update_layout(template='plotly_white', height=250, title=f"Section at {slice_pos:.1f}m", xaxis_title=h_lbl, yaxis_title=v_lbl, margin=dict(t=30,b=0,l=0,r=0))
                st.plotly_chart(fig_sec, use_container_width=True)
            else: st.warning("No points in this slice")

    else: st.error("No Data")

with col_data:
    st.markdown("### 📊 Assessment")
    st.markdown(f"""
    <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
        <small>1. DOH Detection</small><br>Measured: <b>{curr_d:.4f} m</b><br>Rating: <b>{doh}/5</b>
    </div>
    <div class="arrow-box">⬇️</div>
    <div style="border:1px solid #000080; padding:10px; border-radius:5px; background-color:#f0f4ff;">
        <small>2. Advanced Evaluation</small><br>CV: <b>{cv}</b> x W_Comp: <b>{w_c}</b> x W_Defect: <b>{w_d}</b><br><b>Priority: {prio:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""<div style="margin-top:15px; text-align:center;"><span class="{css}">{urg.split('(')[0]}</span><br><h4>{act}</h4></div>""", unsafe_allow_html=True)
    
    # --- UPDATED: DETAILED REFERENCE GUIDE ---
    with st.expander("📘 Reference Standards (คู่มือละเอียด)"):
        t1, t2, t3 = st.tabs(["🇹🇭 DOH Criteria", "🔄 Mapping", "🇪🇺 Algorithm"])
        with t1:
            st.markdown("##### 1. คอนกรีต & โครงสร้าง")
            st.markdown("""
            | Defect | Severity > | Rating | Condition |
            | :--- | :---: | :---: | :--- |
            | **Cracking** | 5.0 mm | **1** | วิกฤต |
            | | 2.0 mm | **2** | รุนแรง |
            | | 0.3 mm | **3** | พอใช้ |
            | **Spalling** | 15 cm | **1** | วิกฤต (เหล็กขาด) |
            | | 10 cm | **2** | รุนแรง (เหล็กสนิม) |
            | | 2.5 cm | **3** | พอใช้ (ถึงเหล็ก) |
            """)
            st.markdown("##### 2. ฐานราก & ความมั่นคง")
            st.markdown("""
            | Defect | Severity > | Rating | Condition |
            | :--- | :---: | :---: | :--- |
            | **Settlement** | 50 mm | **1** | วิกฤต |
            | | 25 mm | **2** | รุนแรง |
            | **Scour** | 50 cm | **1** | วิกฤต (ฐานรากลอย) |
            | | 20 cm | **2** | รุนแรง |
            """)
        with t2: st.write("DOH 5 (Good) -> CV 1 (Good)")
        with t3: 
            st.write("Priority = CV x Comp x Defect")
            st.write("Criticality: > 9.0 (Immediate), 6.0-9.0 (Urgent)")

    with st.form("verify"):
        st.write("---")
        sel_group = st.selectbox("Group", list(BRIDGE_SCHEMA.keys()), index=list(BRIDGE_SCHEMA.keys()).index(item['group']))
        avail_comps = list(BRIDGE_SCHEMA[sel_group].keys())
        sel_comp = st.selectbox("Component", avail_comps, index=avail_comps.index(item['comp']) if item['comp'] in avail_comps else 0)
        avail_defects = BRIDGE_SCHEMA[sel_group][sel_comp]["defects"] + ["No Defect"]
        sel_defect = st.selectbox("Defect", avail_defects, index=avail_defects.index(item['type']) if item['type'] in avail_defects else 0)
        v_depth = st.number_input("Severity (m)", value=float(curr_d), format="%.4f")
        note = st.text_area("Note")
        if st.form_submit_button("💾 Save"):
            st.session_state.results.append({"id":item['id'], "comp":sel_comp, "type":sel_defect, "severity":v_depth, "doh":doh, "priority":prio, "action":act, "note":note, "time":datetime.now().strftime("%Y-%m-%d")})
            st.session_state.idx += 1; st.session_state.manual_depth=None; st.rerun()