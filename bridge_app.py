# --- แก้ไขส่วนนี้เพื่อลดจำนวนจุด (Lighter Version) ---
def generate_complex_structure(defect_type, component_name):
    points_list = []
    
    # ปรับลด Density ลง (จาก 2500 เหลือ 500 และเอาตัวคูณ 5 ออก)
    def add_dense_block(x_lim, y_lim, z_lim, density=500): 
        vol = (x_lim[1]-x_lim[0]) * (y_lim[1]-y_lim[0]) * (z_lim[1]-z_lim[0])
        n_points = int(density * vol) # เอา *5 ออก
        
        # กันเหนียว: ถ้าจุดน้อยเกินไป ให้มีขั้นต่ำ 100 จุด
        if n_points < 100: n_points = 100
        
        # Random Points
        xx = np.random.uniform(x_lim[0], x_lim[1], n_points)
        yy = np.random.uniform(y_lim[0], y_lim[1], n_points)
        zz = np.random.uniform(z_lim[0], z_lim[1], n_points)
        
        # Surface Grid (ลดความละเอียดขอบลงเล็กน้อย)
        xe = np.linspace(x_lim[0], x_lim[1], 15)
        ye = np.linspace(y_lim[0], y_lim[1], 15)
        Xg, Yg = np.meshgrid(xe, ye)
        
        # Add Surfaces
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[0]).flatten()], axis=1))
        points_list.append(np.stack([Xg.flatten(), Yg.flatten(), np.full_like(Xg, z_lim[1]).flatten()], axis=1))
        # Add Volume
        points_list.append(np.stack([xx, yy, zz], axis=1))

    # Dimensions
    L = 12.0; W = 8.0
    z_deck_bot = -0.3; z_girder_bot = -1.5; z_cap_bot = -2.5; z_pier_bot = -6.0

    # === BUILD BRIDGE ===
    # ลด Density รายชิ้นส่วนลง
    add_dense_block([0, L], [0, W], [z_deck_bot, 0], density=800) # Deck
    
    for y in [2.0, 4.0, 6.0]: 
        add_dense_block([0, L], [y-0.3, y+0.3], [z_girder_bot, z_deck_bot], density=600) # Girders
        
    for x in [0.5, L/2, L-0.5]:
        # Diaphragms
        add_dense_block([x-0.15, x+0.15], [2.0, 4.0], [z_girder_bot+0.3, z_deck_bot-0.1], density=400)
        add_dense_block([x-0.15, x+0.15], [4.0, 6.0], [z_girder_bot+0.3, z_deck_bot-0.1], density=400)

    for sx in [2.0, 10.0]:
        add_dense_block([sx-0.6, sx+0.6], [0.5, W-0.5], [z_cap_bot, -1.5], density=800) # Cap
        for py in [2.5, 5.5]:
            add_dense_block([sx-0.4, sx+0.4], [py-0.4, py+0.4], [z_pier_bot, z_cap_bot], density=600) # Pier

    full = np.concatenate(points_list, axis=0)
    X, Y, Z = full[:,0], full[:,1], full[:,2]
    
    # === SIMULATE DEFECTS ===
    Z += np.random.normal(0, 0.005, size=Z.shape)
    ai_depth = 0.0
    
    if defect_type != "No Defect":
        mask = np.zeros_like(Z, dtype=bool)
        if component_name == "Deck" and defect_type == "Spalling":
            mask = (Z > -0.1) & ((X-6)**2 + (Y-4)**2 < 2.5)
            Z[mask] -= 0.15
        elif component_name == "Girder" and defect_type == "Crack":
            mask = (Z < z_girder_bot+0.5) & (abs(Y-4.0)<0.35) & (abs(X-6.0)<0.2)
            Z[mask] += 0.08
        elif component_name == "Cap Beam" and defect_type == "Spalling":
            mask = (abs(X-2.0)<0.7) & (Z>z_cap_bot) & (Y<2.0)
            Z[mask] -= 0.025
            
        if np.any(mask):
            if defect_type == "Spalling": ai_depth = abs(np.min(Z[mask]) - np.mean(Z[~mask]))
            else: ai_depth = np.max(Z[mask]) - np.mean(Z[~mask])

    return X, Y, Z, ai_depth, "Mockup (Balanced)"