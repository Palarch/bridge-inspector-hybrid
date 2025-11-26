import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="Ultimate Bridge Inspector", layout="wide")

# ✨ CSS: CLEAN LIGHT MODE (ปรับให้ดูสะอาดตาแบบรายงานวิศวกรรม)
st.markdown("""
<style>
    /* พื้นหลังหลักเป็นสีขาว (Default) */
    
    /* ปรับแต่งกล่อง Metric ให้ดูนูนขึ้นมานิดนึงและสีสบายตา */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa; /* สีเทาอ่อนมากๆ */
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* ปรับหัวข้อให้ชัดเจน */
    h1, h2, h3 {
        color: #2c3e50; /* สีน้ำเงินเข้มเกือบดำ */
    }
</style>
""", unsafe_allow_html=True)

# --- 2. REALISTIC STRUCTURE GENERATOR (Logic เดิม) ---
def generate_realistic_structure(defect_type):
    points = []
    
    # Dimensions
    length = 10.0
    width = 8.0
    deck_thick = 0.25
    girder_h = 1.0
    girder_w = 0.5
    num_girders = 3
    
    # --- A. Deck ---
    x_d = np.linspace(0, length, 120)
    y_d = np.linspace(0, width, 100)
    X_d, Y_d = np.meshgrid(x_d, y_d)
    Z_top = np.zeros_like(X_d) 
    Z_bot = np.full_like(X_d, -deck_thick)
    
    points.append(np.stack([X_d.flatten(), Y_d.flatten(), Z_top.flatten()], axis=1))
    points.append(np.stack([X_d.flatten(), Y_d.flatten(), Z_bot.flatten()], axis=1))
    
    # --- B. Girders ---
    g_positions = np.linspace(width/(num_girders+1), width - width/(num_girders+1), num_girders)
    
    for gy in g_positions:
        x_g = np.linspace(0, length, 120)
        z_g = np.linspace(-deck_thick, -deck_thick-girder_h, 40)
        X_w, Z_w = np.meshgrid(x_g, z_g)
        Y_w = np.full_like(X_w, gy)
        
        y_f = np.linspace(gy - girder_w/2, gy + girder_w/2, 15)
        X_f, Y_f = np.meshgrid(x_g, y_f)
        Z_f = np.full_like(X_f, -deck_thick-girder_h)
        
        points.append(np.stack([X_w.flatten(), Y_w.flatten(), Z_w.flatten()], axis=1))
        points.append(np.stack([X_f.flatten(), Y_f.flatten(), Z_f.flatten()], axis=1))

    # Combine
    full = np.concatenate(points, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    # --- C. Defects ---
    Z += np.random.normal(0, 0.005, size=Z.shape)
    ai_depth = 0.0
    ai_status = "Safe"
    
    if defect_type == "Spalling":
        mask = (Z > -0.1) & ((X - 5)**2 + (Y - 4)**2 < 4)
        depth_profile = 0.3 * (1 - ((X[mask]-5)**2 + (Y[mask]-4)**2)/4)
        Z[mask] -= depth_profile
        if np.any(mask): ai_depth = abs(np.min(Z[mask]))
        
    elif defect_type == "Crack":
        mid_g = g_positions[1]
        mask = (Z < -1.0) & (abs(Y - mid_g) < 0.3) & (abs(X - 5) < 0.2)
        Z[mask] += 0.1 
        if np.any(mask): ai_depth = np.max(Z[mask]) - (-deck_thick-girder_h)

    if ai_depth > 0.2: ai_status = "Need Repair"
    elif ai_depth > 0.05: ai_status = "Monitor"
        
    return X, Y, Z, ai_depth, ai_status

# --- 3. SESSION STATE ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []
mock_data = [
    {"id": "BR-01", "loc": "Deck Slab (Lane 1)", "type": "Spalling"},
    {"id": "BR-02", "loc": "Girder G2 (Mid-span)", "type": "Crack"},
    {"id": "BR-03", "loc": "Abutment A1", "type": "No Defect"},
]

# --- 4. MAIN INTERFACE ---
st.title("🌉 Ultimate Bridge Inspector")

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

# --- Layout ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📍 {current['loc']}")
    
    # --- Section Tool ---
    st.markdown("##### ✂️ Cross-Section Analyzer")
    slice_pos = st.slider("Station (X-Axis Position)", 0.0, 10.0, 5.0, 0.1)
    
    mask_slice = np.abs(X - slice_pos) < 0.1
    y_slice = Y[mask_slice]
    z_slice = Z[mask_slice]
    
    # 2D Plot (Clean White Theme)
    fig_sec = go.Figure()
    fig_sec.add_trace(go.Scatter(
        x=y_slice, y=z_slice, mode='markers',
        marker=dict(size=5, color=z_slice, colorscale='Jet_r'), # จุดใหญ่ขึ้นนิดนึง
        name='Section Points'
    ))
    fig_sec.update_layout(
        template='plotly_white', # ✨ ธีมขาวสะอาด
        title=f"Section View at Station X = {slice_pos}m",
        xaxis_title="Width (Y)", yaxis_title="Elevation (Z)",
        yaxis_range=[-1.5, 0.2],
        height=280, margin=dict(t=40,b=0,l=0,r=0)
    )
    st.plotly_chart(fig_sec, use_container_width=True)
    
    # 3D Plot (Clean White Theme)
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=2, color=Z, colorscale='Jet_r', showscale=True)
    )])
    plane_y, plane_z = np.meshgrid(np.linspace(0, 8, 10), np.linspace(-1.5, 0.1, 10))
    plane_x = np.full_like(plane_y, slice_pos)
    fig_3d.add_trace(go.Surface(x=plane_x, y=plane_y, z=plane_z, opacity=0.3, colorscale='Reds', showscale=False))
    
    fig_3d.update_layout(
        template='plotly_white', # ✨ ธีมขาวสะอาด
        scene=dict(
            aspectmode='data',
            xaxis=dict(backgroundcolor="white"),
            yaxis=dict(backgroundcolor="white"),
            zaxis=dict(backgroundcolor="white")
        ),
        height=550, margin=dict(t=0,b=0,l=0,r=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with col_right:
    st.markdown("### 🤖 AI Assessment")
    c1, c2 = st.columns(2)
    c1.metric("Est. Severity", f"{ai_d:.3f} m")
    
    color_map = {"Need Repair": "red", "Monitor": "orange", "Safe": "green"}
    status_color = color_map.get(ai_s, "black")
    c2.markdown(f"Status:<br><h3 style='color:{status_color}; margin:0'>{ai_s}</h3>", unsafe_allow_html=True)
    
    st.info("ℹ️ Use the slider to verify the defect depth.")
    
    with st.form("eng_form"):
        st.markdown("### 👷‍♂️ Engineer Verification")
        v_type = st.selectbox("Defect Type", ["Spalling", "Crack", "Corrosion", "No Defect"],
                              index=["Spalling", "Crack", "Corrosion", "No Defect"].index(current['type']))
        v_depth = st.number_input("Confirmed Depth (m)", value=float(ai_d), step=0.001, format="%.3f")
        
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

# --- SIDEBAR ---
st.sidebar.title("Tools")
if st.session_state.results:
    st.sidebar.success(f"Recorded: {len(st.session_state.results)}")
    df_ex = pd.DataFrame(st.session_state.results)
    st.sidebar.download_button("📥 Backup CSV", df_ex.to_csv(index=False).encode('utf-8'), "backup.csv", "text/csv")