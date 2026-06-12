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
def segment_segment_dist_sq_info(p1, p2, q1, q2):
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
        else: sN, sD = -d, a
    elif tN > tD:
        tN = tD
        if (-d+b) < 0.0: sN = 0.0
        elif (-d+b) > a: sN = sD
        else: sN, sD = (-d+b), a
    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD
    diff = (p1 + sc*u) - (q1 + tc*v)
    return np.dot(diff, diff), sc, tc, diff

@njit
def check_collision_pair(pos1, q1, pos2, q2):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    dirs1 = np.zeros((4, 3))
    dirs2 = np.zeros((4, 3))
    for k in range(4): dirs1[k] = q_rotate(q1, orig_dirs[k])
    for k in range(4): dirs2[k] = q_rotate(q2, orig_dirs[k])
    
    for k1 in range(4):
        p1, p2 = pos1, pos1 + dirs1[k1] * LEG_H
        for k2 in range(4):
            q1_s, q2_s = pos2, pos2 + dirs2[k2] * LEG_H
            d_sq, t, u, diff = segment_segment_dist_sq_info(p1, p2, q1_s, q2_s)
            target = (R_BASE + t*(R_TIP-R_BASE)) + (R_BASE + u*(R_TIP-R_BASE))
            if d_sq < (target - 0.002)**2: return True
    return False

def run_interlocked_stacker(target_n=22):
    print(f"Starting SAGE V50 (Interlocked Pair Stacker, {target_n} units)...")
    positions = np.zeros((target_n, 3))
    quaternions = np.zeros((target_n, 4))
    
    # 1. Define the PERFECT NESTING PAIR
    # Unit A: Upright
    pos_a = np.array([0.0, 0.0, 0.5])
    q_a = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Unit B: Inverted and offset
    # Rotation: 180 around Y, 60 around Z
    q_inv = np.array([0.0, 0.0, 1.0, 0.0]) # 180 around Y
    q_rot = np.array([np.cos(np.pi/6), 0.0, 0.0, np.sin(np.pi/6)]) # 60 around Z
    q_b = q_multiply(q_rot, q_inv)
    
    # Find minimum vertical distance for nesting
    pos_b = np.array([0.0, 0.0, 1.3])
    while pos_b[2] > 0.4:
        pos_b[2] -= 0.01
        if check_collision_pair(pos_a, q_a, pos_b, q_b):
            pos_b[2] += 0.015
            break
            
    print(f"  Nesting Pair found with vertical offset: {pos_b[2] - pos_a[2]:.2f}m")
    
    # 2. Pack the Pairs in a 5x2 grid
    idx = 0
    for row in range(2):
        for col in range(5):
            if idx >= target_n: break
            center = np.array([0.8 + col * 1.15, 0.6 + row * 1.15, 0.3])
            # Unit A
            positions[idx] = center + pos_a
            quaternions[idx] = q_a
            idx += 1
            if idx >= target_n: break
            # Unit B
            positions[idx] = center + pos_b
            quaternions[idx] = q_b
            idx += 1
            
    # Remaining
    while idx < target_n:
        positions[idx] = [REAL_CONTAINER[0]/2, REAL_CONTAINER[1]/2, 1.8]
        quaternions[idx] = np.array([1.0, 0, 0, 0])
        idx += 1
        
    return positions, quaternions, target_n

if __name__ == "__main__":
    try:
        pos, qs, count = run_interlocked_stacker(target_n=22)
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
        output_path = "sage_tetrapod/export_results/50_IronContainer_V50_Nesting.glb"
        scene.export(output_path)
        print(f"V50 Result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
