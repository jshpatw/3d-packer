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
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
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
    u = p2 - p1; v = q2 - q1; w = p1 - q1
    a, b, c, d, e = np.dot(u,u), np.dot(u,v), np.dot(v,v), np.dot(u,w), np.dot(v,w)
    D = a*c - b*b
    sN, sD, tN, tD = 0.0, D, 0.0, D
    if D < 1e-8: sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = (b*e - c*d), (a*e - b*d)
        if sN < 0.0: sN = 0.0; tN = e; tD = c
        elif sN > sD: sN = sD; tN = e+b; tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d+b) < 0.0: sN = 0.0
        elif (-d+b) > a: sN = sD
        else: sN = (-d+b); sD = a
    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD
    diff = w + (sc * u) - (tc * v)
    return np.dot(diff, diff), sc, tc

@njit
def check_collision_strict(pos, dirs, packed_pos, packed_dirs, num_packed):
    for i in range(num_packed):
        if np.sum((pos - packed_pos[i])**2) > (LEG_H * 2.8)**2: continue
        for k1 in range(4):
            p1, p2 = pos, pos + dirs[k1] * LEG_H
            for k2 in range(4):
                q1, q2 = packed_pos[i], packed_pos[i] + packed_dirs[i, k2] * LEG_H
                d_sq, t, u = segment_segment_dist_sq(p1, p2, q1, q2)
                target = (R_BASE + t*(R_TIP-R_BASE)) + (R_BASE + u*(R_TIP-R_BASE)) + 0.005
                if d_sq < target**2: return True
    return False

@njit
def is_inside_container(pos, dirs):
    for d in range(3):
        if pos[d] < R_BASE + 0.01 or pos[d] > REAL_CONTAINER[d] - R_BASE - 0.01: return False
    for k in range(4):
        tip = pos + dirs[k] * LEG_H
        for d in range(3):
            if tip[d] < R_TIP + 0.01 or tip[d] > REAL_CONTAINER[d] - R_TIP - 0.01: return False
    return True

def run_v48_exhaustive_stacker(target_n=20):
    print(f"Starting SAGE V48 (Exhaustive Sequential, {target_n} units)...")
    packed_pos = np.zeros((target_n, 3))
    packed_qs = np.zeros((target_n, 4))
    packed_dirs = np.zeros((target_n, 4, 3))
    num_packed = 0
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])

    for i in range(target_n):
        success = False
        # Try 500 random placements
        for attempt in range(500):
            # Biased towards lower Z first, then higher
            z_start = 0.5 + (attempt / 500.0) * (REAL_CONTAINER[2] - 1.0)
            pos = np.array([
                R_BASE + 0.2 + np.random.rand() * (REAL_CONTAINER[0] - 2*R_BASE - 0.4),
                R_BASE + 0.2 + np.random.rand() * (REAL_CONTAINER[1] - 2*R_BASE - 0.4),
                z_start
            ])
            q = np.random.rand(4) - 0.5; q /= np.linalg.norm(q)
            
            dirs = np.zeros((4, 3))
            for k in range(4): dirs[k] = q_rotate(q, orig_dirs[k])
            
            if is_inside_container(pos, dirs) and not check_collision_strict(pos, dirs, packed_pos, packed_dirs, num_packed):
                packed_pos[num_packed] = pos
                packed_qs[num_packed] = q
                packed_dirs[num_packed] = dirs
                num_packed += 1
                print(f"  Unit {num_packed}/{target_n} PLACED (Attempt {attempt+1})")
                success = True
                break
        
        if not success:
            print(f"  Exhaustive search failed for unit {i+1}. Result will be a valid {num_packed} unit pack.")
            break
            
    return packed_pos[:num_packed], packed_qs[:num_packed], num_packed

if __name__ == "__main__":
    try:
        pos, qs, count = run_v48_exhaustive_stacker(target_n=20)
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            matrix = trimesh.transformations.quaternion_matrix(qs[i])
            matrix[:3, 3] = pos[i]
            scene.add_geometry(tp_mesh, transform=matrix, node_name=f"tp_{i}")
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 10]
        scene.add_geometry(container_box, node_name="container")
        output_path = "sage_tetrapod/export_results/48_IronContainer_V48_Exhaustive.glb"
        scene.export(output_path)
        print(f"V48 Result saved to {output_path} | Valid Units: {count}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
