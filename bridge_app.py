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