import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Hybrid Bridge Inspector (Complete)", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding:10px;}
    
    /* Urgency Badges (ตามมาตรฐานรูปที่ 2) */
    .urgency-1 { background-color: #dc3545; color: white; padding: 5px 12px; border-radius: 15px; font-weight: bold; }
    .urgency-2 { background-color: #fd7e14; color: white; padding: 5px 12px; border-radius: 15px; font-weight: bold; }
    .urgency-3 { background-color: #28a745; color: white; padding: 5px 12px; border-radius: 15px; font-weight: bold; }
    
    /* Mapping Arrow */
    .arrow-box { font-size: 20px; text-align: center; margin: 10px 0; color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# --- 2. HYBRID LOGIC ENGINE (Standard Compliance) ---
def calculate_hybrid_assessment(defect_type, measured_depth, component_name):
    # STAGE 1: DOH Detection (5-0 Scale)
    doh_rating = 5 # Default Good
    
    if defect_type != "No Defect":
        if defect_type == "Spalling": # Threshold ตามรูป 4 (>20mm = Rating 1)
            if measured_depth > 0.020: doh_rating = 1
            elif measured_depth > 0.010: doh_rating = 2
            elif measured_depth > 0.005: doh_rating = 3
            else: doh_rating = 4
        elif defect_type == "Crack":
            if measured_depth > 0.005: doh_rating = 1
            elif measured_depth > 0.003: doh_rating = 2
            elif measured_depth > 0.001: doh_rating = 3
            else: doh_rating = 4
        elif defect_type == "Corrosion":
             if measured_depth > 0.05: doh_rating = 2
             else: doh_rating = 3

    # STAGE 2: Mapping (Invert Scale)
    mapping_table = {5:1, 4:2, 3:3, 2:4, 1:5, 0:5}
    cv_score = mapping_table.get(doh_rating, 1)

    # STAGE 3: Evaluation (Pellegrini)
    # Component Weight
    weight = 1.0
    if component_name in ["Girder", "Pier", "Cap Beam"]:
        weight = 1.5 # Primary Member
    
    priority_score = cv_score * weight
    
    # Action Plan
    if priority_score >= 6.0:
        return doh_rating, cv_score, weight, "Urgency 1 (High)", "ซ่อมทันที (Repair Immediately)", "urgency-1"
    elif priority_score >= 3.0:
        return doh_rating, cv_score, weight, "Urgency 2 (Medium)", "ซ่อมระยะสั้น (Short-term Repair)", "urgency-2"
    else:
        return doh_rating, cv_score, weight, "Urgency 3 (Low)", "เฝ้าระวัง (Monitor)", "urgency-3"

# --- 3. DATA HANDLER (Real File OR Mockup) ---
def get_bridge_data(uploaded_file, mock_item):
    if uploaded_file is not None:
        # A. FILE UPLOAD MODE
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower() for c in df.columns]
            if {'x','y','z'}.issubset(df.columns):
                if len(df) > 50000: df = df.sample(50000) # Downsample
                
                X, Y, Z = df['x'].values, df['y'].values, df['z'].values
                # Simple AI for file
                ai_d = abs(np.min(Z) - np.mean(Z)) if len(Z) > 0 else 0
                return X, Y, Z, ai_d, "Real File"
        except: pass

    # B. MOCKUP MODE (Detailed Structure v3.1)
    return generate_complex_structure(mock_item['type'], mock_item['comp'])

# --- 3. STRUCTURE GENERATOR (HIGH DENSITY VERSION) ---
def generate_complex_structure(defect_type, component_name):
    points_list = []
    
    # ฟังก์ชันสร้างก้อนคอนกรีตแบบ "เนื้อแน่น" (Volumetric)
    def add_dense_block(x_lim, y_lim, z_lim, density=3000):
        # คำนวณปริมาตรเพื่อปรับจำนวนจุดให้อัตโนมัติ (ชิ้นใหญ่จุดเยอะ ชิ้นเล็กจุดน้อย)
        vol = (x_lim[1]-x_lim[0]) * (y_lim[1]-y_lim[0]) * (z_lim[1]-z_lim[0])
        n_points = int(density * vol * 5) # คูณ 5 เพื่ออัดจุดให้แน่นปึ้ก
        
        # ป้องกันไม่ให้จุดน้อยเกินไปสำหรับชิ้นส่วนเล็กๆ
        if n_points < 500: n_points = 500
            
        # สุ่มจุดในปริมาตร (Uniform Distribution) ทำให้ดูเป็นเนื้อเดียวกันเหมือนสแกนจริง
        xx = np.random.uniform(x_lim[0], x_lim[1], n_points)
        yy = np.random.uniform(y_lim[0], y_lim[1], n_points)
        zz = np.random.uniform(z_lim[0], z_lim[1], n_points)
        
        # เพิ่มจุดที่ผิวขอบ (Surface) เพื่อให้ขอบคมชัด
        # (สร้างกล่องเปล่าครอบอีกที)
        x_edge = np.linspace(x_lim[0], x_lim[1], 50)
        y_edge = np.linspace(y_lim[0], y_lim[1], 50)
        z_edge = np.linspace(z_lim[0], z_lim[1], 10)
        
        # Top/Bot Surface
        Xg, Yg = np.meshgrid(x_edge, y_edge)
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[0]).flatten()], axis=1))
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[1]).flatten()], axis=1))
        
        # Add Volume Points
        points_list.append(np.stack([xx, yy, zz], axis=1))

    # --- Dimensions (ปรับให้สัดส่วนสมจริงขึ้น) ---
    L = 12.0
    W = 8.0
    
    # ระดับความสูง (Z Levels)
    z_deck_top = 0.0
    z_deck_bot = -0.3   # พื้นหนา 30cm
    z_girder_bot = -1.5 # คานลึก 1.2m
    z_cap_top = -1.5
    z_cap_bot = -2.5    # Cap Beam หนา 1m
    z_pier_bot = -6.0   # เสาสูง 3.5m

    # --- 1. SUPERSTRUCTURE ---
    # A. Deck (พื้นสะพาน) - แผ่นใหญ่
    add_dense_block([0, L], [0, W], [z_deck_bot, z_deck_top], density=4000)
    
    # B. Girders (คานยาว) - 3 ตัว
    girder_y = [2.0, 4.0, 6.0]
    for y in girder_y:
        # คาน I-Shape (จำลองด้วยกล่องสี่เหลี่ยมแต่หนาแน่น)
        add_dense_block([0, L], [y-0.3, y+0.3], [z_girder_bot, z_deck_bot], density=3000)
        
    # C. Diaphragms (คานขวาง)
    dia_x = [0.5, L/2, L-0.5]
    for x in dia_x:
        for i in range(len(girder_y)-1):
            y1, y2 = girder_y[i], girder_y[i+1]
            add_dense_block([x-0.15, x+0.15], [y1+0.3, y2-0.3], [z_girder_bot+0.3, z_deck_bot-0.1])

    # --- 2. SUBSTRUCTURE ---
    support_x = [2.0, 10.0]
    for sx in support_x:
        # D. Cap Beam (คานหัวเสา) - รับ Girder
        add_dense_block([sx-0.6, sx+0.6], [0.5, W-0.5], [z_cap_bot, z_cap_top], density=5000)
        
        # E. Piers (เสาตอม่อ)
        pier_y = [2.5, 5.5]
        for py in pier_y:
            add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_pier_bot, z_cap_bot], density=4000)

    # Combine All
    full_cloud = np.concatenate(points_list, axis=0)
    X, Y, Z = full_cloud[:,0], full_cloud[:,1], full_cloud[:,2]
    
    # --- 3. DEFECT SIMULATION ---
    # Add Noise (Texture)
    Z += np.random.normal(0, 0.005, size=Z.shape)
    
    ai_depth = 0.0
    if defect_type != "No Defect":
        mask = np.zeros_like(Z, dtype=bool)
        
        if component_name == "Deck" and defect_type == "Spalling":
            # หลุมบนพื้น
            mask = (Z > z_deck_top - 0.1) & ((X-6)**2 + (Y-4)**2 < 2.5)
            Z[mask] -= 0.15 # หลุมลึก
            
        elif component_name == "Girder" and defect_type == "Crack":
            # รอยร้าวกลางคานตัวกลาง
            mask = (Z < z_girder_bot + 0.5) & (abs(Y-4.0) < 0.35) & (abs(X-6.0) < 0.2)
            Z[mask] += 0.08 # นูนออกมา
            
        elif component_name == "Cap Beam" and defect_type == "Spalling":
            # แตกที่หัวคาน Cap Beam
            mask = (abs(X-2.0) < 0.7) & (Z > z_cap_bot) & (Y < 2.0)
            Z[mask] -= 0.2
            
        # Calculate AI Depth
        if np.any(mask):
            if defect_type == "Spalling":
                ai_depth = abs(np.min(Z[mask]) - np.mean(Z[~mask]))
            else:
                ai_depth = np.max(Z[mask]) - np.mean(Z[~mask])

    return X, Y, Z, ai_depth, "Mockup (High-Res)"
# --- 4. UI SETUP ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []

mock_data = [
    {"id":"C1", "comp":"Deck", "group":"Superstructure", "type":"Spalling", "loc":"Mid-span"},
    {"id":"C2", "comp":"Girder", "group":"Superstructure", "type":"Crack", "loc":"G2 Beam"},
    {"id":"C3", "comp":"Cap Beam", "group":"Substructure", "type":"Spalling", "loc":"Pier 1 Cap"},
]

# Sidebar
st.sidebar.title("🛠️ Tools")
uploaded_file = st.sidebar.file_uploader("Upload Point Cloud (.csv)", type=['csv'])
if st.session_state.results:
    df_ex = pd.DataFrame(st.session_state.results)
    st.sidebar.download_button("📥 Backup CSV", df_ex.to_csv(index=False).encode('utf-8'), "backup.csv", "text/csv")

# Main Page
st.title("🌉 Hybrid Bridge Inspector v5.0")
st.caption("Standards: DOH Detection ➔ Pellegrini Management | Feature: 3D/2D Analysis + File Support")

if st.session_state.idx >= len(mock_data):
    st.success("Inspection Batch Completed!")
    if st.session_state.results: st.dataframe(pd.DataFrame(st.session_state.results))
    if st.button("Restart"): st.session_state.idx=0; st.session_state.results=[]; st.rerun()
    st.stop()

# ... (ส่วนบนปล่อยไว้เหมือนเดิม เริ่มแก้ตรง Load Data ลงไป) ...

# Load Data
item = mock_data[st.session_state.idx]
X, Y, Z, ai_depth, source = get_bridge_data(uploaded_file, item)

# Calculate Hybrid Rating
doh, cv, w, urgency, action, css = calculate_hybrid_assessment(item['type'], ai_depth, item['comp'])

# --- LAYOUT DEFINITION (จุดที่ Error) ---
# ต้องประกาศตัวแปร col_viz และ col_data ก่อนเรียกใช้
col_viz, col_data = st.columns([1.8, 1]) 

with col_viz:
    st.subheader(f"📍 {item['comp']} ({source})")
    
    # 1. SECTION SLIDER & 2D GRAPH
    st.markdown("##### ✂️ Cross-Section Analyzer")
    # ป้องกัน Error กรณี X เป็น None หรือว่าง
    if len(X) > 0:
        min_x, max_x = float(np.min(X)), float(np.max(X))
        slice_pos = st.slider("Station (X-Axis)", min_x, max_x, (min_x+max_x)/2)
        
        mask_slice = np.abs(X - slice_pos) < 0.2
        y_slice = Y[mask_slice]; z_slice = Z[mask_slice]
        
        fig_sec = go.Figure(go.Scatter(x=y_slice, y=z_slice, mode='markers', marker=dict(size=4, color=z_slice, colorscale='Jet_r')))
        fig_sec.update_layout(template='plotly_white', height=250, title=f"Section at X={slice_pos:.1f}m", yaxis_title="Z", xaxis_title="Y", margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_sec, use_container_width=True)

       # 2. 3D VISUALIZATION
    # ปรับ size=1.5 หรือ 2 และ opacity=0.8 ให้ดูมีมิติ
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z, 
        mode='markers', 
        marker=dict(size=1.5, color=Z, colorscale='Jet_r', opacity=0.8) 
    )])

with col_data:
    st.markdown("### 📊 Assessment")
    
    # DOH CARD
    st.markdown(f"""
    <div style="border:1px solid #ddd; padding:10px; border-radius:5px;">
        <small>Stage 1: Detection (DOH)</small><br>
        Measured: <b>{ai_depth*1000:.1f} mm</b><br>
        Rating: <b>{doh} / 5</b>
    </div>
    <div class="arrow-box">⬇️ Invert Mapping ⬇️</div>
    """, unsafe_allow_html=True)
    
    # PELLEGRINI CARD
    st.markdown(f"""
    <div style="border:1px solid #000080; padding:10px; border-radius:5px; background-color:#f0f4ff;">
        <small>Stage 3: Management (Pellegrini)</small><br>
        CV Score: <b>{cv}</b> (1=Good, 5=Fail)<br>
        Weight: <b>x{w}</b> ({item['comp']})<br>
        <hr style="margin:5px 0">
        Priority Score: <b>{cv*w:.1f}</b>
    </div>
    """, unsafe_allow_html=True)
    
    # FINAL ACTION
    st.markdown(f"""
    <div style="margin-top:15px; text-align:center;">
        <span class="{css}">{urgency}</span><br>
        <h4>{action}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("verify"):
        st.write("---")
        v_type = st.selectbox("Type", ["Spalling", "Crack", "Corrosion", "No Defect"], index=["Spalling", "Crack", "Corrosion", "No Defect"].index(item['type']))
        v_depth = st.number_input("Confirmed Depth (m)", value=float(ai_depth), format="%.3f")
        note = st.text_area("Note")
        
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id":item['id'], "comp":item['comp'], "doh":doh, "cv":cv, "urgency":urgency, "note":note,
                "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.idx += 1
            st.rerun()