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
    u, v, w = p2 - p1, q2 - q1, p1 - q1
    a, b, c, d, e = np.dot(u,u), np.dot(u,v), np.dot(v,v), np.dot(u,w), np.dot(v,w)
    D = a*c - b*b
    sN, sD, tN, tD = 0.0, D, 0.0, D
    if D < 1e-8: sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = (b*e - c*d), (a*e - b*d)
        if sN < 0.0: sN, tN, tD = 0.0, e, c
        elif sN > sD: sN, tN, tD = sD, e+b, c
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
def hard_resolve_collisions(positions, quaternions, n, iterations=200):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    all_dirs = np.zeros((n, 4, 3))
    
    for _it in range(iterations):
        for i in range(n):
            for k in range(4): all_dirs[i, k] = q_rotate(quaternions[i], orig_dirs[k])

        max_overlap = 0.0
        # 1. HARD OBJECT RESOLVE
        for i in range(n):
            for j in range(i + 1, n):
                if np.sum((positions[i] - positions[j])**2) > (LEG_H * 2.8)**2: continue
                for k1 in range(4):
                    p1, p2 = positions[i], positions[i] + all_dirs[i, k1] * LEG_H
                    for k2 in range(4):
                        q1, q2 = positions[j], positions[j] + all_dirs[j, k2] * LEG_H
                        d_sq, t, u, diff = segment_segment_dist_sq_info(p1, p2, q1, q2)
                        target = (R_BASE + t*(R_TIP-R_BASE)) + (R_BASE + u*(R_TIP-R_BASE)) + 0.005
                        if d_sq < target**2:
                            dist = np.sqrt(d_sq)
                            overlap = target - dist
                            max_overlap = max(max_overlap, overlap)
                            push = (diff / (dist + 1e-9)) * (overlap * 0.51) # Over-push slightly
                            positions[i] += push
                            positions[j] -= push

        # 2. HARD CONTAINER RESOLVE
        for i in range(n):
            for d in range(3):
                if positions[i, d] < R_BASE: positions[i, d] = R_BASE
                if positions[i, d] > REAL_CONTAINER[d] - R_BASE: positions[i, d] = REAL_CONTAINER[d] - R_BASE
            for k in range(4):
                point = positions[i] + all_dirs[i, k] * LEG_H
                rad = R_TIP
                for d in range(3):
                    if point[d] < rad: positions[i, d] += (rad - point[d])
                    if point[d] > REAL_CONTAINER[d] - rad: positions[i, d] -= (point[d] - (REAL_CONTAINER[d] - rad))
        
        if max_overlap < 0.001: break
    return max_overlap

def run_v46_hard_stacker(target_n=22):
    print(f"Starting SAGE V46 (Hard Geometry Resolver, {target_n} units)...")
    positions = np.zeros((target_n, 3))
    quaternions = np.zeros((target_n, 4))
    
    # Grid Layout
    idx = 0
    for layer in range(2):
        for row in range(2):
            for col in range(5):
                if idx >= target_n: break
                positions[idx] = [0.8 + col * 1.1, 0.6 + row * 1.1, 0.7 + layer * 1.1]
                quaternions[idx] = np.array([1.0, 0, 0, 0]) if layer == 0 else np.array([0.0, 0, 1.0, 0])
                idx += 1
    while idx < target_n:
        positions[idx] = [REAL_CONTAINER[0]/2, REAL_CONTAINER[1]/2, 1.2]
        quaternions[idx] = np.array([1.0, 0, 0, 0]); idx += 1

    print("  Initial layout set. Starting Hard Resolution...")
    final_overlap = hard_resolve_collisions(positions, quaternions, target_n, iterations=500)
    print(f"  Final Max Overlap: {final_overlap:.6f}m")
    
    return positions, quaternions, target_n

if __name__ == "__main__":
    try:
        pos, qs, count = run_v46_hard_stacker(target_n=22)
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
        output_path = "sage_tetrapod/export_results/46_IronContainer_V46_Hard.glb"
        scene.export(output_path)
        print(f"V46 Result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
