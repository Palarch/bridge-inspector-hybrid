import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="Bridge Inspector AI", layout="wide")

# CSS ปรับแต่ง UI
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .reportview-container .main .block-container{ padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA GENERATOR & AI LOGIC ---
def generate_point_cloud_and_measure(defect_type):
    """
    สร้าง Point Cloud และให้ AI ลองวัดขนาด (Simulate AI Measurement)
    """
    # Create Grid
    x = np.linspace(0, 10, 80)
    y = np.linspace(0, 10, 80)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) # Reference Plane
    
    # Add Noise
    Z += np.random.normal(0, 0.005, size=X.shape)

    # Generate Defect Shape
    if defect_type == "Spalling":
        mask = (X - 5)**2 + (Y - 5)**2 < 4
        depth_profile = 0.5 * (1 - ((X - 5)**2 + (Y - 5)**2)/4)
        Z[mask] -= depth_profile[mask]
        
    elif defect_type == "Crack":
        mask = np.abs(X - Y) < 0.25
        Z[mask] -= 0.15 
        
    elif defect_type == "Corrosion":
        mask = (X > 3) & (X < 7) & (Y > 3) & (Y < 7)
        Z[mask] += np.random.normal(0.05, 0.02, size=np.sum(mask))

    # --- AI AUTO-MEASUREMENT LOGIC ---
    # AI จะสแกนหาจุดที่ลึกที่สุด (Min Z) หรือจุดที่นูนที่สุด (Max Z)
    raw_min = np.min(Z)
    raw_max = np.max(Z)
    
    # คำนวณความรุนแรง (Severity) แบบคร่าวๆ
    ai_measured_depth = 0.0
    if abs(raw_min) > abs(raw_max):
        ai_measured_depth = abs(raw_min) # กรณีเป็นหลุม/รอยแตก
    else:
        ai_measured_depth = abs(raw_max) # กรณีบวม
        
    # AI Suggest Status (Logic เบื้องต้น)
    ai_status_suggestion = "Safe"
    if ai_measured_depth > 0.3:
        ai_status_suggestion = "Need Repair"
    elif ai_measured_depth > 0.1:
        ai_status_suggestion = "Monitor"

    return X.flatten(), Y.flatten(), Z.flatten(), ai_measured_depth, ai_status_suggestion

# --- 3. SESSION STATE ---
if 'defect_index' not in st.session_state: st.session_state.defect_index = 0
if 'results' not in st.session_state: st.session_state.results = []
if 'mock_data' not in st.session_state:
    st.session_state.mock_data = [
        {"id": "D-001", "loc": "Girder G1", "type": "Spalling"},
        {"id": "D-002", "loc": "Deck Slab", "type": "Crack"},
        {"id": "D-003", "loc": "Pier P2", "type": "Corrosion"},
    ]

# --- 4. MAIN UI ---
st.title("🌉 Bridge Inspector: AI-Hybrid Mode")
st.progress((st.session_state.defect_index) / len(st.session_state.mock_data))

# Check Finish
if st.session_state.defect_index >= len(st.session_state.mock_data):
    st.success("✅ Inspection Completed!")
    if len(st.session_state.results) > 0:
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Final Report CSV", csv, "final_report.csv", "text/csv")
    if st.button("Restart"):
        st.session_state.defect_index = 0
        st.session_state.results = []
        st.rerun()
    st.stop()

# Load Current Item
current_item = st.session_state.mock_data[st.session_state.defect_index]
X, Y, Z, ai_depth, ai_status = generate_point_cloud_and_measure(current_item['type'])

# Layout
col_viz, col_form = st.columns([1.8, 1])

# --- LEFT COLUMN: 3D Visualization & Cross Section ---
with col_viz:
    st.subheader(f"📍 Location: {current_item['loc']}")
    
    # Section Slider
    st.markdown("##### 🔍 Engineer Cross-Check Tool")
    slice_pos = st.slider("Move Section Plane (X-Axis)", 0.0, 10.0, 5.0, 0.1)
    
    # Calculate Section Data
    mask_slice = np.abs(X - slice_pos) < 0.2
    y_slice = Y[mask_slice]
    z_slice = Z[mask_slice]
    
    # 2D Plot
    fig_sec = go.Figure()
    fig_sec.add_trace(go.Scatter(x=y_slice, y=z_slice, mode='markers', marker=dict(color='red', size=4)))
    fig_sec.update_layout(
        title=f"Section View at X={slice_pos}m", 
        height=200, margin=dict(l=0,r=0,t=30,b=0),
        yaxis_title="Depth (m)", xaxis_title="Width (m)", yaxis_range=[-0.6, 0.1]
    )
    st.plotly_chart(fig_sec, use_container_width=True)
    
    # 3D Plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=2, color=Z, colorscale='Viridis', showscale=True)
    )])
    # Add Red Plane
    plane_x = np.full((10, 10), slice_pos)
    plane_y, plane_z = np.meshgrid(np.linspace(0, 10, 10), np.linspace(-0.6, 0.1, 10))
    fig_3d.add_trace(go.Surface(x=plane_x, y=plane_y, z=plane_z, opacity=0.3, colorscale='Reds', showscale=False))
    fig_3d.update_layout(scene=dict(aspectmode='data'), height=400, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_3d, use_container_width=True)

# --- RIGHT COLUMN: AI Report & Verification Form ---
with col_form:
    st.markdown("### 🤖 AI Analysis Report")
    
    # AI Estimation Card
    with st.container():
        c1, c2 = st.columns(2)
        c1.metric("Est. Max Depth", f"{ai_depth:.3f} m")
        
        # Color Logic for Status
        status_color = "red" if ai_status == "Need Repair" else "orange" if ai_status == "Monitor" else "green"
        c2.markdown(f"**AI Suggestion:**")
        c2.markdown(f":{status_color}[{ai_status}]")
    
    st.info("ℹ️ The AI has auto-measured the deepest point from the scan. Please verify with the Section Tool.")
    
    st.markdown("---")
    st.markdown("### 👷‍♂️ Engineer Verification")
    
    with st.form("inspection_form"):
        # 1. Confirm Type
        ver_type = st.selectbox("Defect Type", ["Spalling", "Crack", "Corrosion", "No Defect"], 
                                index=["Spalling", "Crack", "Corrosion", "No Defect"].index(current_item['type']))
        
        # 2. Confirm Severity (Pre-filled with AI value)
        ver_depth = st.number_input("Confirmed Depth/Width (m)", value=float(ai_depth), step=0.001, format="%.3f")
        
        # 3. STATUS BUTTONS (Restored!)
        st.write("**Assessment Status:**")
        # Pre-select based on AI suggestion
        status_options = ["Safe", "Monitor", "Need Repair"]
        default_idx = status_options.index(ai_status)
        ver_status = st.radio("Choose Status:", status_options, index=default_idx, horizontal=True)
        
        # 4. Comments
        comments = st.text_area("Engineering Notes", placeholder="Additional observations...")
        
        # Submit
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id": current_item['id'],
                "location": current_item['loc'],
                "verified_type": ver_type,
                "verified_depth": ver_depth,
                "final_status": ver_status, # เก็บค่าสถานะ
                "comments": comments,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.defect_index += 1
            st.rerun()

# Sidebar Export
st.sidebar.title("Data Tools")
if len(st.session_state.results) > 0:
    st.sidebar.success(f"Recorded: {len(st.session_state.results)}")
    df_ex = pd.DataFrame(st.session_state.results)
    csv_ex = df_ex.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download CSV", csv_ex, "latest_inspection.csv", "text/csv")