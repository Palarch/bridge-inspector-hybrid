import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="BridgeDefect Inspector Pro", layout="wide")

# CSS ปรับแต่งให้สวยงาม
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED DATA GENERATOR (With Depth Profile) ---
def generate_mock_point_cloud(defect_type):
    # สร้าง Grid ถี่ขึ้นเพื่อให้เห็น Section ชัดเจน
    x = np.linspace(0, 10, 80)
    y = np.linspace(0, 10, 80)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) # Reference Plane
    
    # Noise ผิวคอนกรีต
    Z += np.random.normal(0, 0.005, size=X.shape)

    # สร้างความเสียหายที่มี Volume จริง
    if defect_type == "Spalling":
        # หลุมกว้าง
        mask = (X - 5)**2 + (Y - 5)**2 < 4
        # สูตรถ้วย (Parabola) ให้ตรงกลางลึกสุด
        depth_profile = 0.5 * (1 - ((X - 5)**2 + (Y - 5)**2)/4)
        Z[mask] -= depth_profile[mask] 
        
    elif defect_type == "Crack":
        # รอยร้าวแคบแต่ลึก
        mask = np.abs(X - Y) < 0.25
        Z[mask] -= 0.15 # ลึก 15 cm
        
    elif defect_type == "Corrosion":
        # สนิมบวม (Pitting)
        mask = (X > 3) & (X < 7) & (Y > 3) & (Y < 7)
        Z[mask] += np.random.normal(0.05, 0.02, size=np.sum(mask))

    return X.flatten(), Y.flatten(), Z.flatten()

# --- 3. SESSION STATE ---
if 'defect_index' not in st.session_state: st.session_state.defect_index = 0
if 'results' not in st.session_state: st.session_state.results = []
if 'mock_data' not in st.session_state:
    st.session_state.mock_data = [
        {"id": "D-001", "loc": "Girder G1", "ai_type": "Spalling", "ai_sev": 15.0, "conf": 0.91},
        {"id": "D-002", "loc": "Deck Slab", "ai_type": "Crack", "ai_sev": 0.45, "conf": 0.85},
        {"id": "D-003", "loc": "Pier P2", "ai_type": "No Defect", "ai_sev": 0.0, "conf": 0.40},
        {"id": "D-004", "loc": "Bearing B1", "ai_type": "Corrosion", "ai_sev": 5.0, "conf": 0.65},
    ]

# --- 4. MAIN INTERFACE ---
st.title("🌉 Bridge Inspection: Hybrid Interface")
st.markdown("**AI Detection + Engineering Depth Analysis**")

# Progress Bar
progress = (st.session_state.defect_index) / len(st.session_state.mock_data)
st.progress(progress)

# Check Completion
if st.session_state.defect_index >= len(st.session_state.mock_data):
    st.success("🎉 Inspection Completed! Please download the report from the sidebar.")
    if st.button("Start New Batch"):
        st.session_state.defect_index = 0
        st.session_state.results = []
        st.rerun()
    st.stop()

# Load Data
current_defect = st.session_state.mock_data[st.session_state.defect_index]
X, Y, Z = generate_mock_point_cloud(current_defect['ai_type'])

# Layout: 2 Columns
col_viz, col_ctrl = st.columns([2, 1])

# --- LEFT: VISUALIZATION & MEASUREMENT ---
with col_viz:
    st.subheader(f"📍 {current_defect['loc']} (ID: {current_defect['id']})")
    
    # 1. Measurement Slider
    st.markdown("##### 📏 Depth Analyzer (Section A-A)")
    slice_pos = st.slider("Move Cutting Plane (X-Axis)", 0.0, 10.0, 5.0, 0.1, help="Slide to see cross-section profile")
    
    # Logic: Cross-section Calculation
    mask_slice = np.abs(X - slice_pos) < 0.2
    y_slice = Y[mask_slice]
    z_slice = Z[mask_slice]
    
    max_depth_val = 0.0
    if len(z_slice) > 0:
        max_depth_val = abs(np.min(z_slice))
    
    # 2. 2D Cross-section Plot
    fig_section = go.Figure()
    fig_section.add_trace(go.Scatter(
        x=y_slice, y=z_slice, mode='markers',
        marker=dict(size=4, color='red'), name='Surface'
    ))
    fig_section.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Surface Level")
    fig_section.update_layout(
        title=f"Section View at X = {slice_pos:.1f} m",
        yaxis_title="Depth (m)", xaxis_title="Width (m)",
        yaxis_range=[-0.6, 0.2], height=250, margin=dict(t=30, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_section, use_container_width=True)
    
    # 3. 3D Point Cloud Plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=2, color=Z, colorscale='Viridis', showscale=True, colorbar=dict(title="Depth"))
    )])
    
    # Add Cutting Plane Visualization
    plane_x = np.full((10, 10), slice_pos)
    plane_y, plane_z = np.meshgrid(np.linspace(0, 10, 10), np.linspace(-0.6, 0.1, 10))
    fig_3d.add_trace(go.Surface(x=plane_x, y=plane_y, z=plane_z, opacity=0.3, colorscale='Reds', showscale=False))
    
    fig_3d.update_layout(scene=dict(aspectmode='data'), height=450, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_3d, use_container_width=True)

# --- RIGHT: VERIFICATION FORM ---
with col_ctrl:
    st.markdown("### 👷‍♂️ Engineer Panel")
    
    # AI Info
    st.info(f"🤖 AI Prediction:\n\n**{current_defect['ai_type']}** ({current_defect['conf']*100:.0f}% Conf.)")
    
    # Measurement Result
    st.metric("Measured Max Depth", f"{max_depth_val:.3f} m")
    
    st.markdown("---")
    
    with st.form("verify_form"):
        st.write("**Verification Decision**")
        
        # Pre-fill data
        def_idx = ['Spalling', 'Crack', 'Corrosion', 'No Defect'].index(current_defect['ai_type']) if current_defect['ai_type'] in ['Spalling', 'Crack', 'Corrosion', 'No Defect'] else 3
        
        final_type = st.selectbox("Defect Type", ['Spalling', 'Crack', 'Corrosion', 'No Defect'], index=def_idx)
        final_sev = st.number_input("Severity Value", value=float(current_defect['ai_sev']))
        
        # ใช้ค่าจากการวัด (Optional)
        use_measured = st.checkbox("Use Measured Depth as Severity")
        if use_measured:
            final_sev = max_depth_val
            st.caption(f"Will save: {final_sev:.3f}")
            
        note = st.text_area("Notes", placeholder="Engineering judgement...")
        
        submitted = st.form_submit_button("💾 Save & Next", type="primary")
        
        if submitted:
            # Save Data Logic
            st.session_state.results.append({
                "id": current_defect['id'],
                "location": current_defect['loc'],
                "ai_type": current_defect['ai_type'],
                "final_type": final_type,
                "final_severity": final_sev,
                "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.defect_index += 1
            st.rerun()

# --- 5. EXPORT SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("Tools")
st.sidebar.markdown("---")

if len(st.session_state.results) > 0:
    st.sidebar.success(f"✅ Recorded: {len(st.session_state.results)} items")
    df_export = pd.DataFrame(st.session_state.results)
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "📥 Download CSV Report",
        csv,
        f"inspection_report_{datetime.now().strftime('%H%M')}.csv",
        "text/csv"
    )
else:
    st.sidebar.info("Pending Inspection...")

st.sidebar.markdown("---")
st.sidebar.caption("Hybrid Bridge Inspection System v1.2")