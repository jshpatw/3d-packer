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
    sN, sD = 0.0, D; tN, tD = 0.0, D
    if D < 1e-8: sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = (b*e - c*d), (a*e - b*d)
        if sN < 0.0: sN = 0.0; tN = e; tD = c
        elif sN > sD: sN = sD; tN = e + b; tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN = (-d + b); sD = a
    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD
    diff = w + (sc * u) - (tc * v)
    return np.dot(diff, diff), sc, tc

@njit
def get_total_overlap_v62(pos, qs, n):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    all_dirs = np.zeros((n, 4, 3))
    for i in range(n):
        for k in range(4): all_dirs[i, k] = q_rotate(qs[i], orig_dirs[k])
    
    total = 0.0
    for i in range(n):
        for d in range(3):
            if pos[i, d] < R_BASE: total += (R_BASE - pos[i, d]) * 10.0
            if pos[i, d] > REAL_CONTAINER[d] - R_BASE: total += (pos[i, d] - (REAL_CONTAINER[d] - R_BASE)) * 10.0
        for k in range(4):
            tip = pos[i] + all_dirs[i, k] * LEG_H
            for d in range(3):
                if tip[d] < R_TIP: total += (R_TIP - tip[d]) * 10.0
                if tip[d] > REAL_CONTAINER[d] - R_TIP: total += (tip[d] - (REAL_CONTAINER[d] - R_TIP)) * 10.0
        for j in range(i + 1, n):
            if np.sum((pos[i] - pos[j])**2) > (LEG_H * 3.2)**2: continue
            for k1 in range(4):
                p1, p2 = pos[i], pos[i] + all_dirs[i, k1] * LEG_H
                for k2 in range(4):
                    q1, q2 = pos[j], pos[j] + all_dirs[j, k2] * LEG_H
                    d_sq, sc, tc = segment_segment_dist_sq(p1, p2, q1, q2)
                    target = (R_BASE + sc*(R_TIP-R_BASE)) + (R_BASE + tc*(R_TIP-R_BASE))
                    if d_sq < target**2:
                        total += (target - np.sqrt(d_sq))
    return total

@njit
def run_monte_carlo_v62(pos, qs, n, steps=200000, temp_start=0.1):
    current_overlap = get_total_overlap_v62(pos, qs, n)
    for s in range(steps):
        i = np.random.randint(0, n)
        old_p = pos[i].copy(); old_q = qs[i].copy()
        
        progress = s/steps
        temp = temp_start * (1.0 - progress)**2 + 0.001
        
        pos[i] += (np.random.rand(3) - 0.5) * temp
        dq = (np.random.rand(4) - 0.5) * temp * 4.0
        qs[i] = (qs[i] + dq) / np.linalg.norm(qs[i] + dq)
        
        new_overlap = get_total_overlap_v62(pos, qs, n)
        if new_overlap < current_overlap:
            current_overlap = new_overlap
        elif np.random.rand() < np.exp(-(new_overlap - current_overlap) / (temp * 0.1)):
            current_overlap = new_overlap
        else:
            pos[i] = old_p; qs[i] = old_q
        if current_overlap < 1e-6: return current_overlap, s
    return current_overlap, steps

def run_22unit_variation(variation_id):
    target_n = 22
    print(f"Generating 22-unit Variation {variation_id}...")
    
    positions = np.zeros((target_n, 3))
    quaternions = np.zeros((target_n, 4))
    
    # Randomly select a base topology
    topo_type = variation_id % 3
    idx = 0
    if topo_type == 0:
        # Standard staggered
        for layer in range(2):
            for row in range(2):
                for col in range(6 if row == 0 else 5):
                    if idx >= target_n: break
                    positions[idx] = [0.6 + col * 0.95, 0.6 + row * 1.15, 0.6 + layer * 1.15]
                    quaternions[idx] = [1,0,0,0]
                    idx += 1
    elif topo_type == 1:
        # Diagonal sweep
        for layer in range(2):
            for i in range(11):
                positions[idx] = [0.6 + (i%4)*1.3 + (i//4)*0.4, 0.6 + (i%3)*0.6, 0.6 + layer * 1.2]
                quaternions[idx] = [1,0,0,0]
                idx += 1
    else:
        # Random clusters
        positions = np.random.rand(target_n, 3) * (REAL_CONTAINER - 1.2) + 0.6
        for i in range(target_n):
            q = np.random.rand(4) - 0.5
            quaternions[idx] = q / np.linalg.norm(q)
            idx += 1

    steps_per_block = 50000
    total_steps = 1000000
    curr_ov = get_total_overlap_v62(positions, quaternions, target_n)
    
    for block in range(total_steps // steps_per_block):
        # Adaptive cooling
        progress = (block * steps_per_block) / total_steps
        temp = 0.15 * (1.0 - progress)**2 + 0.001
        curr_ov, s = run_monte_carlo_v62(positions, quaternions, target_n, steps=steps_per_block, temp_start=temp)
        print(f"    Var {variation_id} Block {block+1:2d} | Overlap: {curr_ov:.4f}m | T: {temp:.4f}")
        if curr_ov < 1e-6: break

    return positions, quaternions, target_n, curr_ov

if __name__ == "__main__":
    try:
        import generate_mesh
        for v in range(3): # Generate 3 distinct variations
            pos, qs, count, ov = run_22unit_variation(v)
            if ov < 0.1: # Only save if reasonably successful
                scene = trimesh.Scene()
                for i in range(count):
                    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
                    matrix = trimesh.transformations.quaternion_matrix(qs[i])
                    matrix[:3, 3] = pos[i]
                    scene.add_geometry(tp_mesh, transform=matrix, node_name=f"tp_{i}")
                container_box = trimesh.creation.box(extents=REAL_CONTAINER)
                container_box.apply_translation(REAL_CONTAINER/2)
                scene.add_geometry(container_box, node_name="container")
                output_path = f"sage_tetrapod/export_results/01_Safe_Packs/62_Var_{v}_22Unit.glb"
                scene.export(output_path)
                print(f"Variation {v} saved to {output_path} | Overlap: {ov:.6f}m")
            else:
                print(f"Variation {v} failed to converge (Overlap: {ov:.4f}m). Skipping export.")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
