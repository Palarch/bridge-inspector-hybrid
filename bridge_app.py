import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="BridgeDefect Inspector", layout="wide")

# CSS ปรับแต่งเล็กน้อยให้ดูสะอาดตา
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .stButton>button { width: 100%; }
    .big-font { font-size:20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. MOCK DATA GENERATOR (Point Cloud) ---
def generate_mock_point_cloud(defect_type):
    # สร้างข้อมูลจำลอง (เหมือนเดิม)
    x = np.linspace(0, 10, 60)
    y = np.linspace(0, 10, 60)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    X_f, Y_f, Z_f = X.flatten(), Y.flatten(), Z.flatten()
    colors = np.zeros_like(Z_f)
    
    # Noise พื้นฐาน
    Z_f += np.random.normal(0, 0.02, size=Z_f.shape)

    if defect_type == "Spalling":
        mask = (X_f - 5)**2 + (Y_f - 5)**2 < 5
        Z_f[mask] -= np.random.uniform(0.1, 0.4, size=np.sum(mask))
        colors[mask] = 1 
    elif defect_type == "Crack":
        mask = np.abs(X_f - Y_f) < 0.4
        Z_f[mask] -= 0.08
        colors[mask] = 1

    return X_f, Y_f, Z_f, colors

# --- 3. SESSION STATE MANAGEMENT ---
# ใช้สำหรับจำค่าระหว่างการกดปุ่ม (เพราะเว็บจะ refresh ทุกครั้งที่มีการ interact)
if 'defect_index' not in st.session_state:
    st.session_state.defect_index = 0
if 'results' not in st.session_state:
    st.session_state.results = []
if 'mock_data' not in st.session_state:
    # จำลองข้อมูลตั้งต้น 5 รายการ
    st.session_state.mock_data = [
        {"id": "D-001", "loc": "Girder G1", "ai_type": "Crack", "ai_sev": 0.45, "conf": 0.91},
        {"id": "D-002", "loc": "Abutment A1", "ai_type": "Spalling", "ai_sev": 12.5, "conf": 0.78},
        {"id": "D-003", "loc": "Deck Slab", "ai_type": "No Defect", "ai_sev": 0.0, "conf": 0.40},
        {"id": "D-004", "loc": "Pier P3", "ai_type": "Crack", "ai_sev": 0.30, "conf": 0.85},
        {"id": "D-005", "loc": "Bearing B2", "ai_type": "Corrosion", "ai_sev": 5.0, "conf": 0.60},
    ]

# --- 4. MAIN INTERFACE ---

# HEADER
st.title("🌉 Hybrid Bridge Inspection Platform")
st.markdown("**(AI-Assisted Defect Verification System)**")
st.progress((st.session_state.defect_index) / len(st.session_state.mock_data))

# CHECK IF FINISHED
if st.session_state.defect_index >= len(st.session_state.mock_data):
    st.success("✅ Inspection Batch Completed!")
    st.write("### Summary Report")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)
    
    # ปุ่ม Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report (CSV)", csv, "inspection_report.csv", "text/csv")
    
    if st.button("Start New Batch"):
        st.session_state.defect_index = 0
        st.session_state.results = []
        st.rerun()
    st.stop() # หยุดการทำงานส่วนล่าง

# LOAD CURRENT DATA
current_defect = st.session_state.mock_data[st.session_state.defect_index]

# LAYOUT: 2 COLUMNS
col_left, col_right = st.columns([2, 1])

# --- LEFT COLUMN: 3D VISUALIZATION ---
with col_left:
    st.subheader(f"📍 Location: {current_defect['loc']}")
    
    # Generate 3D Plot
    x, y, z, c = generate_mock_point_cloud(current_defect['ai_type'])
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=3, color=c, colorscale='RdYlGn_r', opacity=0.8) # เขียวไปแดง
    )])
    fig.update_layout(scene=dict(aspectmode='data', zaxis=dict(range=[-1, 1])), height=500, margin=dict(l=0,r=0,b=0,t=0))
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Use mouse to Rotate, Zoom, and Pan the Point Cloud model.")

# --- RIGHT COLUMN: ENGINEER CONTROLS ---
with col_right:
    st.info(f"🔢 Defect ID: {current_defect['id']}")
    
    # AI Proposal Card
    st.markdown("### 🤖 AI Analysis")
    c1, c2 = st.columns(2)
    c1.metric("Detected Type", current_defect['ai_type'])
    c2.metric("Confidence", f"{current_defect['conf']*100:.0f}%")
    
    st.markdown("---")
    
    # Human Verification Form
    st.markdown("### 👷‍♂️ Engineer Verification")
    
    with st.form("verification_form"):
        # Auto-fill ค่าจาก AI
        verified_type = st.selectbox(
            "Defect Type", 
            ['Crack', 'Spalling', 'Corrosion', 'Leaching', 'No Defect'],
            index=['Crack', 'Spalling', 'Corrosion', 'Leaching', 'No Defect'].index(current_defect['ai_type']) if current_defect['ai_type'] in ['Crack', 'Spalling', 'Corrosion', 'Leaching', 'No Defect'] else 4
        )
        
        severity = st.number_input("Severity (width/area)", value=float(current_defect['ai_sev']), step=0.1)
        
        status = st.radio("Status", ["Wait for Repair", "Monitor", "Safe"], horizontal=True)
        
        notes = st.text_area("Engineering Comment", placeholder="Add specific observations...")
        
        # Submit Button
        submitted = st.form_submit_button("💾 Verify & Next")
        
        if submitted:
            # Save Data
            record = {
                "id": current_defect['id'],
                "original_ai_type": current_defect['ai_type'],
                "verified_type": verified_type,
                "severity": severity,
                "status": status,
                "notes": notes,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.results.append(record)
            
            # Next Item
            st.session_state.defect_index += 1
            st.rerun()

# --- SIDEBAR INFO ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.markdown("## Project Controls")
st.sidebar.markdown(f"**Inspector:** Eng. Student")
st.sidebar.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About Hybrid Mode"):
    st.write("""
    This system combines **AI Detection** from Point Cloud data with **Human Expertise** for final verification.
    """)