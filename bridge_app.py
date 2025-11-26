import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="Bridge Inspector Final", layout="wide")

# CSS: Clean Light Mode
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 8px;
    }
    h1, h2, h3 { color: #2c3e50; }
    .stApp { background-color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED STRUCTURE GENERATOR (With Super/Sub Structure) ---
def generate_complex_bridge(defect_type, component_group, component_name):
    points = []
    
    # Helper: สร้างกล่องสี่เหลี่ยม
    def add_box(x_range, y_range, z_range, density=12):
        xx = np.linspace(x_range[0], x_range[1], int((x_range[1]-x_range[0])*density + 3))
        yy = np.linspace(y_range[0], y_range[1], int((y_range[1]-y_range[0])*density + 3))
        zz = np.linspace(z_range[0], z_range[1], int((z_range[1]-z_range[0])*density + 3))
        
        # Top/Bottom
        X, Y = np.meshgrid(xx, yy)
        points.append(np.stack([X.flatten(), Y.flatten(), np.full_like(X, z_range[0]).flatten()], axis=1))
        points.append(np.stack([X.flatten(), Y.flatten(), np.full_like(X, z_range[1]).flatten()], axis=1))
        # Sides
        X, Z = np.meshgrid(xx, zz)
        points.append(np.stack([X.flatten(), np.full_like(X, y_range[0]).flatten(), Z.flatten()], axis=1))
        points.append(np.stack([X.flatten(), np.full_like(X, y_range[1]).flatten(), Z.flatten()], axis=1))
        # Ends
        Y, Z = np.meshgrid(yy, zz)
        points.append(np.stack([np.full_like(Y, x_range[0]).flatten(), Y.flatten(), Z.flatten()], axis=1))
        points.append(np.stack([np.full_like(Y, x_range[1]).flatten(), Y.flatten(), Z.flatten()], axis=1))

    # Dimensions
    bridge_len = 12.0; bridge_width = 8.0
    z_deck_top=0.0; z_deck_bot=-0.25; z_girder_bot=-1.25
    z_cap_top=-1.25; z_cap_bot=-2.0; z_pier_bot=-5.0
    
    num_girders = 3
    girder_width = 0.5
    girder_spacing = bridge_width / (num_girders + 1)
    girder_y_centers = [girder_spacing * (i+1) for i in range(num_girders)]
    support_x_locs = [2.0, 10.0] 

    # --- Build Components ---
    # 1. Deck
    add_box([0, bridge_len], [0, bridge_width], [z_deck_bot, z_deck_top])
    # 2. Girders
    for gy in girder_y_centers:
        add_box([0, bridge_len], [gy-0.25, gy+0.25], [z_girder_bot, z_deck_bot])
    # 3. Diaphragms
    for dx in [0.5, bridge_len/2, bridge_len-0.5]:
        for i in range(num_girders - 1):
            add_box([dx-0.15, dx+0.15], [girder_y_centers[i]+0.25, girder_y_centers[i+1]-0.25], [z_girder_bot+0.2, z_deck_bot-0.2])
    # 4. Cap Beam & Piers
    for sx in support_x_locs:
        add_box([sx-0.4, sx+0.4], [0.5, bridge_width-0.5], [z_cap_bot, z_cap_top]) # Cap
        for py in [bridge_width*0.3, bridge_width*0.7]:
            add_box([sx-0.3, sx+0.3], [py-0.3, py+0.3], [z_pier_bot, z_cap_bot]) # Pier

    full_cloud = np.concatenate(points, axis=0)
    X, Y, Z = full_cloud[:,0], full_cloud[:,1], full_cloud[:,2]
    
    # --- Defects ---
    Z += np.random.normal(0, 0.005, size=Z.shape)
    ai_depth = 0.0
    ai_status = "Safe"
    mask = np.zeros_like(Z, dtype=bool)
    
    if defect_type != "No Defect":
        if component_name == "Deck":
            mask = (Z > z_deck_top - 0.05) & ((X - 6)**2 + (Y - 4)**2 < 2)
            if defect_type == "Spalling": Z[mask] -= 0.15
            elif defect_type == "Crack": Z[mask] -= 0.05
        elif component_name == "Girder":
            mask = (Z < z_girder_bot + 0.2) & (abs(Y - girder_y_centers[1]) < 0.25) & (abs(X - 6) < 0.2)
            if defect_type == "Crack": Z[mask] += 0.1
        elif component_name == "Cap Beam":
            mask = (abs(X - support_x_locs[0]) < 0.5) & (Z > z_cap_bot) & (Y < 1.0)
            if defect_type == "Spalling": Z[mask] -= 0.2

        if np.any(mask):
            ai_depth = np.max(Z[mask]) - np.mean(Z[~mask]) if defect_type == "Crack" else abs(np.min(Z[mask]) - np.mean(Z[~mask]))
    
    if ai_depth > 0.15: ai_status = "Need Repair"
    elif ai_depth > 0.05: ai_status = "Monitor"

    return X, Y, Z, ai_depth, ai_status

# --- 3. SESSION ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'results' not in st.session_state: st.session_state.results = []

mock_data = [
    {"id": "SP-01", "group": "Superstructure", "comp": "Deck", "type": "Spalling", "loc": "Mid-span"},
    {"id": "SP-02", "group": "Superstructure", "comp": "Girder", "type": "Crack", "loc": "G2 Mid-span"},
    {"id": "SP-03", "group": "Superstructure", "comp": "Diaphragm", "type": "No Defect", "loc": "End Diaphragm"},
    {"id": "SB-01", "group": "Substructure", "comp": "Cap Beam", "type": "Spalling", "loc": "Pier Cap 1"},
    {"id": "SB-02", "group": "Substructure", "comp": "Pier", "type": "No Defect", "loc": "Pier 1"},
]

# --- 4. UI ---
st.title("🌉 Bridge Inspector Final (v3.1)")

if st.session_state.idx >= len(mock_data):
    st.success("✅ Completed!")
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df)
        st.download_button("📥 Final CSV", df.to_csv(index=False).encode('utf-8'), "final.csv", "text/csv")
    if st.button("Restart"): st.session_state.idx = 0; st.session_state.results = []; st.rerun()
    st.stop()

item = mock_data[st.session_state.idx]
X, Y, Z, ai_d, ai_s = generate_complex_bridge(item['type'], item['group'], item['comp'])

# Layout
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📍 Checking: {item['comp']} ({item['group']})")
    
    # --- 1. SECTION TOOL (RESTORED!) ---
    st.markdown("##### ✂️ Cross-Section Analyzer")
    slice_pos = st.slider("Station (X-Axis)", 0.0, 12.0, 6.0, 0.1)
    
    # Calculate Section
    mask_slice = np.abs(X - slice_pos) < 0.15
    y_slice = Y[mask_slice]
    z_slice = Z[mask_slice]
    
    # 2D Section Plot
    fig_sec = go.Figure()
    fig_sec.add_trace(go.Scatter(
        x=y_slice, y=z_slice, mode='markers',
        marker=dict(size=4, color=z_slice, colorscale='Jet_r'),
    ))
    fig_sec.update_layout(
        template='plotly_white', height=250, title=f"Section at X={slice_pos}m",
        yaxis_title="Elevation (Z)", xaxis_title="Width (Y)", margin=dict(t=30,b=0,l=0,r=0)
    )
    st.plotly_chart(fig_sec, use_container_width=True)
    
    # --- 2. 3D MODEL ---
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers',
        marker=dict(size=2, color=Z, colorscale='Jet_r', showscale=True)
    )])
    # Cutting Plane
    py, pz = np.meshgrid(np.linspace(0, 8, 10), np.linspace(-5, 0, 10))
    px = np.full_like(py, slice_pos)
    fig_3d.add_trace(go.Surface(x=px, y=py, z=pz, opacity=0.3, colorscale='Reds', showscale=False))
    
    fig_3d.update_layout(
        template='plotly_white', height=500, margin=dict(t=0,b=0,l=0,r=0),
        scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)))
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with col_right:
    st.markdown("### 🤖 AI Assessment")
    c1, c2 = st.columns(2)
    c1.metric("Severity", f"{ai_d:.3f} m")
    col = "red" if ai_s == "Need Repair" else "orange" if ai_s == "Monitor" else "green"
    c2.markdown(f"Status:<br><h3 style='color:{col}'>{ai_s}</h3>", unsafe_allow_html=True)
    
    with st.form("verify"):
        st.markdown("### 👷‍♂️ Verification")
        v_type = st.selectbox("Type", ["Spalling", "Crack", "Corrosion", "No Defect"], 
                              index=["Spalling", "Crack", "Corrosion", "No Defect"].index(item['type']))
        v_depth = st.number_input("Confirmed Depth", value=float(ai_d), format="%.3f")
        v_status = st.radio("Status", ["Safe", "Monitor", "Need Repair"], 
                            index=["Safe", "Monitor", "Need Repair"].index(ai_s), horizontal=True)
        note = st.text_area("Notes")
        
        if st.form_submit_button("💾 Save & Next", type="primary"):
            st.session_state.results.append({
                "id": item['id'], "comp": item['comp'], "group": item['group'],
                "type": v_type, "depth": v_depth, "status": v_status, "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.idx += 1
            st.rerun()

if st.session_state.results:
    st.sidebar.download_button("📥 Backup CSV", pd.DataFrame(st.session_state.results).to_csv(index=False).encode('utf-8'), "backup.csv", "text/csv")