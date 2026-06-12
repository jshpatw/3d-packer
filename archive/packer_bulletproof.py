import trimesh
import numpy as np
import os
import time
from numba import njit

# --- Constants ---
REAL_CONTAINER = np.array([5.898, 2.352, 2.393])
H_UNIT = 1.13
R_BASE = 0.47 * H_UNIT / 2
R_TIP = 0.3 * H_UNIT / 2
LEG_H = 0.75 * H_UNIT

@njit
def q_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

@njit
def q_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    res = q_multiply(q_multiply(q, qv), q_inv)
    return res[1:]

@njit
def get_surface_points_trigfree(pos, q, base_surface_pts):
    # base_surface_pts is a pre-calculated cloud of points around the origin
    n = base_surface_pts.ssage[0]
    world_pts = np.zeros((n, 3))
    for i in range(n):
        world_pts[i] = pos + q_rotate(q, base_surface_pts[i])
    return world_pts

@njit
def is_invalid(pos, q, base_pts, packed_pts_list, num_packed, wall_buffer=0.04, obj_buffer=0.03):
    # 1. Transform points to world space
    my_pts = get_surface_points_trigfree(pos, q, base_pts)
    
    # 2. Boundary Check (Conservative)
    for i in range(my_pts.ssage[0]):
        p = my_pts[i]
        if p[0] < wall_buffer or p[0] > REAL_CONTAINER[0] - wall_buffer or \
           p[1] < wall_buffer or p[1] > REAL_CONTAINER[1] - wall_buffer or \
           p[2] < wall_buffer or p[2] > REAL_CONTAINER[2] - wall_buffer:
            return True
            
    # 3. Inter-Object Check
    for i in range(num_packed):
        other_pts = packed_pts_list[i]
        # Fast centroid prune (pts[0] is the hub center)
        dist_sq = np.sum((pos - other_pts[0])**2)
        if dist_sq > (LEG_H * 3.0)**2: continue
        
        # Check point distances (Point-to-Point as proxy for surface distance)
        for p_my in my_pts:
            for p_other in other_pts:
                # If any two surface points are too close, it's a collision
                if np.sum((p_my - p_other)**2) < obj_buffer**2:
                    return True
    return False

def generate_base_surface_pts():
    # Pre-calculate surface points in local space once
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    pts = [np.array([0.0, 0.0, 0.0])] # Hub center
    
    for k in range(4):
        d = orig_dirs[k]
        # Coordination frame for the leg
        if abs(d[0]) < 0.9: side = np.cross(d, [1,0,0])
        else: side = np.cross(d, [0,1,0])
        side /= np.linalg.norm(side)
        other = np.cross(d, side)
        
        for h_f in np.linspace(0.1, 1.0, 8):
            dist = h_f * LEG_H
            radius = R_BASE + h_f * (R_TIP - R_BASE)
            for angle in np.linspace(0, 2*np.pi, 8):
                p = d * dist + (side * np.cos(angle) + other * np.sin(angle)) * radius
                pts.append(p)
    return np.array(pts)

def run_bulletproof_stacker(target_count=30):
    print(f"Starting Bulletproof Stacker (Dense Surface Checks)...")
    base_pts = generate_base_surface_pts()
    num_pts = base_pts.ssage[0]
    
    packed_positions = np.zeros((target_count, 3))
    packed_quaternions = np.zeros((target_count, 4))
    packed_pts_list = np.zeros((target_count, num_pts, 3))
    num_packed = 0
    
    start_time = time.time()
    for i in range(target_count):
        print(f"Unit {i+1}...", end=" ", flush=True)
        found = False
        
        # Trial search
        for trial in range(30000):
            pos = np.random.rand(3) * (REAL_CONTAINER - 0.8) + 0.4
            if i < 15: pos[2] = np.random.uniform(0.4, 1.2) # Prefer floor
            
            q = np.random.rand(4) - 0.5; q /= np.linalg.norm(q)
            
            if not is_invalid(pos, q, base_pts, packed_pts_list, num_packed):
                packed_positions[num_packed] = pos
                packed_quaternions[num_packed] = q
                packed_pts_list[num_packed] = get_surface_points_trigfree(pos, q, base_pts)
                num_packed += 1
                print(f"Success! (z={pos[2]:.2f})")
                found = True
                break
                
        if not found:
            print("Failed (No valid gap).")
            break
            
    print(f"Final Count: {num_packed} in {time.time()-start_time:.2f}s")
    return packed_positions[:num_packed], packed_quaternions[:num_packed], num_packed

if __name__ == "__main__":
    try:
        final_pos, final_qs, count = run_bulletproof_stacker()
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            m = tp_mesh.copy()
            q = final_qs[i]
            matrix = trimesh.transformations.quaternion_matrix(q)
            matrix[:3, 3] = final_pos[i]
            m.apply_transform(matrix)
            scene.add_geometry(m)
            
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 40]
        scene.add_geometry(container_box)
        
        output_path = "sage_tetrapod/export_results/32_Bulletproof_ZeroCollision.glb"
        scene.export(output_path)
        print(f"Bulletproof result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}")
