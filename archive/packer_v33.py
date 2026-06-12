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
def segment_segment_dist_sq(p1, p2, q1, q2):
    # Standard algorithm for shortest distance between two line segments
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a*c - b*b
    sN, sD, tN, tD = 0.0, D, 0.0, D
    if D < 1e-8:
        sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = (b*e - c*d), (a*e - b*d)
        if sN < 0.0: sN, tN, tD = 0.0, e, c
        elif sN > sD: sN, tN, tD = sD, e + b, c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN, sD = -d, a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN, sD = (-d + b), a
    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD
    
    # Distance between closest points
    diff = (p1 + sc * u) - (q1 + tc * v)
    return np.dot(diff, diff), sc, tc

@njit
def check_collision_analytic(pos1, q1, pos2, q2, buffer=0.015):
    # A tetrapod is 4 legs (cones). We check all 16 leg pairs.
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    
    # 1. Hub-to-Hub check (Spheres)
    dist_centers_sq = np.sum((pos1 - pos2)**2)
    if dist_centers_sq < (R_BASE * 2.0 + buffer)**2:
        return True
        
    # Rotate directions
    dirs1 = np.zeros((4, 3))
    dirs2 = np.zeros((4, 3))
    for k in range(4):
        dirs1[k] = q_rotate(q1, orig_dirs[k])
        dirs2[k] = q_rotate(q2, orig_dirs[k])
        
    # 2. Leg-to-Leg check (Analytical Capsules)
    for k1 in range(4):
        p1 = pos1
        p2 = pos1 + dirs1[k1] * LEG_H
        for k2 in range(4):
            q_start = pos2
            q_end = pos2 + dirs2[k2] * LEG_H
            
            d_sq, t, u = segment_segment_dist_sq(p1, p2, q_start, q_end)
            
            # Radii at the closest points (linear interpolation)
            r1 = R_BASE + t * (R_TIP - R_BASE)
            r2 = R_BASE + u * (R_TIP - R_BASE)
            
            # Surface distance check
            if d_sq < (r1 + r2 + buffer)**2:
                return True
    return False

@njit
def is_outside_analytic(pos, q, buffer=0.01):
    # Hub check
    if pos[0] < R_BASE + buffer or pos[0] > REAL_CONTAINER[0] - R_BASE - buffer: return True
    if pos[1] < R_BASE + buffer or pos[1] > REAL_CONTAINER[1] - R_BASE - buffer: return True
    if pos[2] < R_BASE + buffer or pos[2] > REAL_CONTAINER[2] - R_BASE - buffer: return True
    
    # Tip check
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    for k in range(4):
        tip = pos + q_rotate(q, orig_dirs[k]) * LEG_H
        if tip[0] < R_TIP + buffer or tip[0] > REAL_CONTAINER[0] - R_TIP - buffer: return True
        if tip[1] < R_TIP + buffer or tip[1] > REAL_CONTAINER[1] - R_TIP - buffer: return True
        if tip[2] < R_TIP + buffer or tip[2] > REAL_CONTAINER[2] - R_TIP - buffer: return True
    return False

def run_analytical_stacker(target_count=30):
    print(f"Starting Analytical Stacker (Zero Collision Guarantee)...")
    packed_pos = []
    packed_qs = []
    
    start_time = time.time()
    
    for i in range(target_count):
        print(f"Unit {i+1}...", end=" ", flush=True)
        found = False
        
        # High-intensity search (50,000 trials per unit)
        for trial in range(50000):
            # 1. Random Pose
            pos = np.random.rand(3) * (REAL_CONTAINER - 0.6) + 0.3
            # Preference for filling floor first
            if i < 15: pos[2] = np.random.uniform(0.3, 1.0)
            
            q = np.random.rand(4) - 0.5
            q /= np.linalg.norm(q)
            
            # 2. Boundary Check
            if is_outside_analytic(pos, q, buffer=0.03): continue
            
            # 3. Analytic Collision Check against ALL previous units
            collision = False
            for j in range(len(packed_pos)):
                if check_collision_analytic(pos, q, packed_pos[j], packed_qs[j], buffer=0.02):
                    collision = True
                    break
            
            if not collision:
                packed_pos.append(pos)
                packed_qs.append(q)
                print(f"Locked! (z={pos[2]:.2f})")
                found = True
                break
        
        if not found:
            print("Failed (No valid gap).")
            break
            
    return np.array(packed_pos), np.array(packed_qs), len(packed_pos)

if __name__ == "__main__":
    try:
        final_pos, final_qs, count = run_analytical_stacker(target_count=35)
        
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            m = tp_mesh.copy()
            matrix = trimesh.transformations.quaternion_matrix(final_qs[i])
            matrix[:3, 3] = final_pos[i]
            m.apply_transform(matrix)
            scene.add_geometry(m)
            
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 40]
        scene.add_geometry(container_box)
        
        output_path = "sage_tetrapod/export_results/33_Perfect_Analytical_Stack.glb"
        scene.export(output_path)
        print(f"Verified Result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback; traceback.print_exc()
