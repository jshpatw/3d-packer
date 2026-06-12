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
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

@njit
def segment_segment_dist_sq_info(p1, p2, q1, q2):
    u, v, w = p2 - p1, q2 - q1, p1 - q1
    a, b, c, d, e = np.dot(u, u), np.dot(u, v), np.dot(v, v), np.dot(u, w), np.dot(v, w)
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
    diff = (p1 + sc * u) - (q1 + tc * v)
    return np.dot(diff, diff), sc, tc, diff

@njit
def resolve_constraints_sequential(pos, q, packed_pos, packed_dirs, num_packed, iterations=30):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    
    for _it in range(iterations):
        my_dirs = np.zeros((4, 3))
        for k in range(4): my_dirs[k] = q_rotate(q, orig_dirs[k])
        
        # 1. Collision with Packed
        for i in range(num_packed):
            if np.sum((pos - packed_pos[i])**2) > (LEG_H * 2.8)**2: continue
            for k1 in range(4):
                p1, p2 = pos, pos + my_dirs[k1] * LEG_H
                for k2 in range(4):
                    q1, q2 = packed_pos[i], packed_pos[i] + packed_dirs[i, k2] * LEG_H
                    d_sq, t, u, diff = segment_segment_dist_sq_info(p1, p2, q1, q2)
                    target = (R_BASE + t*(R_TIP-R_BASE)) + (R_BASE + u*(R_TIP-R_BASE)) + 0.01
                    if d_sq < target**2:
                        dist = np.sqrt(d_sq)
                        push = (diff / (dist + 1e-9)) * (target - dist)
                        pos += push
                        
                        lever = (p1 + t*(p2-p1)) - pos
                        torque = cross(lever, push)
                        dq = 0.5 * q_multiply(q, np.array([0.0, torque[0], torque[1], torque[2]])) * 0.5
                        q += dq
                        q /= np.linalg.norm(q)
                        for k in range(4): my_dirs[k] = q_rotate(q, orig_dirs[k])

        # 2. Container
        for d in range(3):
            if pos[d] < R_BASE: pos[d] = R_BASE
            if pos[d] > REAL_CONTAINER[d] - R_BASE: pos[d] = REAL_CONTAINER[d] - R_BASE
        for k in range(4):
            point = pos + my_dirs[k] * LEG_H
            radius = R_TIP
            for d in range(3):
                if point[d] < radius: pos[d] += (radius - point[d])
                if point[d] > REAL_CONTAINER[d] - radius: pos[d] -= (point[d] - (REAL_CONTAINER[d] - radius))
    return pos, q

def run_hybrid_sequential(target_n=22):
    print(f"Starting SAGE V42 (Sequential Hybrid, {target_n} units)...")
    packed_pos = np.zeros((target_n, 3))
    packed_qs = np.zeros((target_n, 4))
    packed_dirs = np.zeros((target_n, 4, 3))
    num_packed = 0
    
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])

    for i in range(target_n):
        # Try multiple random starting drops
        best_pos = None
        best_q = None
        lowest_z = 999.0
        
        for attempt in range(10):
            pos = np.array([
                R_BASE + np.random.rand() * (REAL_CONTAINER[0] - 2*R_BASE),
                R_BASE + np.random.rand() * (REAL_CONTAINER[1] - 2*R_BASE),
                REAL_CONTAINER[2] - R_BASE - 0.5
            ])
            q = np.random.rand(4) - 0.5
            q /= np.linalg.norm(q)
            
            # Physics drop
            for step in range(1500):
                # Gravity
                pos[2] -= 0.015
                # Resolve
                pos, q = resolve_constraints_sequential(pos, q, packed_pos, packed_dirs, num_packed, iterations=2)
                # Jitter
                if step < 1000:
                    pos += (np.random.rand(3) - 0.5) * 0.01
            
            if pos[2] < lowest_z:
                lowest_z = pos[2]
                best_pos = pos.copy()
                best_q = q.copy()
        
        packed_pos[num_packed] = best_pos
        packed_qs[num_packed] = best_q
        for k in range(4): packed_dirs[num_packed, k] = q_rotate(best_q, orig_dirs[k])
        num_packed += 1
        print(f"  Packed {num_packed}/{target_n} at Z={best_pos[2]:.2f}")
        
    return packed_pos, packed_qs, num_packed

if __name__ == "__main__":
    try:
        pos, qs, count = run_hybrid_sequential(target_n=22)
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
        output_path = "sage_tetrapod/export_results/42_IronContainer_V42_Sequential.glb"
        scene.export(output_path)
        print(f"V42 Result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
