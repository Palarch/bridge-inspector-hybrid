import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="Ultimate Bridge Inspector", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #262730; border: 1px solid #464b59; padding: 10px; border-radius: 5px; }
    /* ปรับแต่ง Slider ให้เห็นชัด */
    div[data-baseweb="slider"] { padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. REALISTIC STRUCTURE GENERATOR (Deck + Girders) ---
def generate_realistic_structure(defect_type):
    points = []
    
    # Dimensions
    length = 10.0   # ความยาวสะพาน (X)
    width = 8.0     # ความกว้าง (Y)
    deck_thick = 0.25
    girder_h = 1.0
    girder_w = 0.5
    num_girders = 3
    
    # --- A. สร้าง Deck (แผ่นพื้น) ---
    x_d = np.linspace(0, length, 120)
    y_d = np.linspace(0, width, 100)
    X_d, Y_d = np.meshgrid(x_d, y_d)
    
    # ผิวบน (Top)
    Z_top = np.zeros_like(X_d) 
    # ผิวล่าง (Bottom)
    Z_bot = np.full_like(X_d, -deck_thick)
    
    points.append(np.stack([X_d.flatten(), Y_d.flatten(), Z_top.flatten()], axis=1))
    points.append(np.stack([X_d.flatten(), Y_d.flatten(), Z_bot.flatten()], axis=1))
    
    # --- B. สร้าง Girders (คานรับน้ำหนัก) ---
    g_positions = np.linspace(width/(num_girders+1), width - width/(num_girders+1), num_girders)
    
    for gy in g_positions:
        # Web (เอวคาน)
        x_g = np.linspace(0, length, 120)
        z_g = np.linspace(-deck_thick, -deck_thick-girder_h, 40)
        X_w, Z_w = np.meshgrid(x_g, z_g)
        Y_w = np.full_like(X_w, gy) # อยู่ตรงกลาง
        
        # Flange (ปีกคานล่าง)
        y_f = np.linspace(gy - girder_w/2, gy + girder_w/2, 15)
        X_f, Y_f = np.meshgrid(x_g, y_f)
        Z_f = np.full_like(X_f, -deck_thick-girder_h)
        
        points.append(np.stack([X_w.flatten(), Y_w.flatten(), Z_w.flatten()], axis=1))
        points.append(np.stack([X_f.flatten(), Y_f.flatten(), Z_f.flatten()], axis=1))

    # Combine All
    full = np.concatenate(points, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    # --- C. ใส่ความเสียหาย (Defects) ---
    # Add Noise
    Z += np.random.normal(0, 0.005, size=Z.shape)
    
    ai_depth = 0.0
    ai_status = "Safe"
    
    if defect_type == "Spalling":
        # หลุมบนพื้น Deck
        mask = (Z > -0.1) & ((X - 5)**2 + (Y - 4)**2 < 4)
        depth_profile = 0.3 * (1 - ((X[mask]-5)**2 + (Y[mask]-4)**2)/4)
        Z[mask] -= depth_profile
        # AI Logic
        if np.any(mask): ai_depth = abs(np.min(Z[mask]))
        
    elif defect_type == "Crack":
        # รอยร้าวที่คานตัวกลาง
        mid_g = g_positions[1]
        mask = (Z < -1.0) & (abs(Y - mid_g) < 0.3) & (abs(X - 5) < 0.2)
        Z[mask] += 0.1 # Crack นูน/แตกออกมา
        # AI Logic
        if np.any(mask): ai_depth = np.max(Z[mask]) - (-deck_thick-girder_h)

    # Status Logic
    if ai_depth > 0.2: ai_status = "Need Repair"
    elif ai_depth > 0.05: ai_status = "Monitor"
        
    return X, Y, Z, ai_depth, ai_status

# --- 3. SESSION STATE ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []
# Mock Data
mock_data = [
    {"id": "BR-01", "loc": "Deck Slab (Lane 1)", "type": "Spalling"},
    {"id": "BR-02", "loc": "Girder G2 (Mid-span)", "type": "Crack"},
    {"id": "BR-03", "loc": "Abutment A1", "type": "No Defect"},
]

# --- 4. MAIN INTERFACE ---
st.title("🌉 Ultimate Bridge Inspector")

# Progress
if st.session_state.idx >= len(mock_data):
    st.success("✅ Inspection Completed!")
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df)
        st.download_button("📥 Final CSV", df.to_csv(index=False).encode('utf-8'), "final.csv", "text/csv")
    if st.button("Restart"):
        st.session_state.idx = 0; st.session_state.results = []; st.rerun()
    st.stop()

current = mock_data[st.session_state.idx]
X, Y, Z, ai_d, ai_s = generate_realistic_structure(current['type'])

# --- Layout: 2 Columns ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📍 {current['loc']}")
    
    # --- SECTION TOOL (The Missing Piece Restored!) ---
    st.markdown("##### ✂️ Cross-Section Analyzer")
    # Slider ตัดตามแนวแกน X (ความยาวสะพาน) เพื่อดูหน้าตัด Y-Z
    slice_pos = st.slider("Station (X-Axis Position)", 0.0, 10.0, 5.0, 0.1)
    
    # Slicing Logic
    # เลือกจุดที่อยู่ในระยะ +/- 0.1 เมตร จากตำแหน่ง Slider
    mask_slice = np.abs(X - slice_pos) < 0.1
    y_slice = Y[mask_slice]
    z_slice = Z[mask_slice]
    
    # 2D Cross-Section Plot
    fig_sec = go.Figure()
    fig_sec.add_trace(go.Scatter(
        x=y_slice, y=z_slice, mode='markers',
        marker=dict(size=4, color=z_slice, colorscale='Jet_r'),
        name='Section Points'
    ))
    fig_sec.update_layout(
        title=f"Section View at Station X = {slice_pos}m",
        xaxis_title="Width (Y)", yaxis_title="Elevation (Z)",
        yaxis_range=[-1.5, 0.2], # Fix แกนให้เห็นคานครบ
        height=250, margin=dict(t=30,b=0,l=0,r=0)
    )
    st.plotly_chart(fig_sec, use_container_width=True)
    
    # 3D Plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=1.5, color=Z, colorscale='Jet_r', showscale=True)
    )])
    # Add Red Cutting Plane
    plane_y, plane_z = np.meshgrid(np.linspace(0, 8, 10), np.linspace(-1.5, 0.1, 10))
    plane_x = np.full_like(plane_y, slice_pos)
    fig_3d.add_trace(go.Surface(x=plane_x, y=plane_y, z=plane_z, opacity=0.3, colorscale='Reds', showscale=False))
    
    fig_3d.update_layout(scene=dict(aspectmode='data'), height=500, margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig_3d, use_container_width=True)

with col_right:
    st.markdown("### 🤖 AI Assessment")
    c1, c2 = st.columns(2)
    c1.metric("Est. Severity", f"{ai_d:.3f} m")
    color = "red" if ai_s == "Need Repair" else "orange" if ai_s == "Monitor" else "green"
    c2.markdown(f"Status:<br><strong style='color:{color}'>{ai_s}</strong>", unsafe_allow_html=True)
    
    st.info("ℹ️ Use the slider to verify the defect depth in the Cross-Section view.")
    
    with st.form("eng_form"):
        st.markdown("### 👷‍♂️ Engineer Verification")
        v_type = st.selectbox("Defect Type", ["Spalling", "Crack", "Corrosion", "No Defect"],
                              index=["Spalling", "Crack", "Corrosion", "No Defect"].index(current['type']))
        v_depth = st.number_input("Confirmed Depth (m)", value=float(ai_d), step=0.001, format="%.3f")
        
        # Status Buttons
        st.write("Final Status:")
        st_opts = ["Safe", "Monitor", "Need Repair"]
        v_status = st.radio("Select:", st_opts, index=st_opts.index(ai_s), horizontal=True)
        
        note = st.text_area("Notes")
        
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id": current['id'], "loc": current['loc'],
                "type": v_type, "depth": v_depth, "status": v_status, "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.idx += 1
            st.rerun()

# --- SIDEBAR EXPORT ---
st.sidebar.title("Tools")
if st.session_state.results:
    st.sidebar.success(f"Recorded: {len(st.session_state.results)}")
    df_ex = pd.DataFrame(st.session_state.results)
    st.sidebar.download_button("📥 Backup CSV", df_ex.to_csv(index=False).encode('utf-8'), "backup.csv", "text/csv")