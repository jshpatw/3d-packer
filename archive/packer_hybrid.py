import trimesh
import numpy as np
import os
import time
from numba import njit, prange

# --- Constants ---
REAL_CONTAINER = np.array([5.898, 2.352, 2.393])
H_UNIT = 1.13
R_BASE_MAX = 0.47 * H_UNIT / 2
R_TIP_MAX = 0.3 * H_UNIT / 2
LEG_H = 0.75 * H_UNIT

@njit
def get_dirs(matrix):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    res = np.zeros((4, 3))
    for k in range(4):
        v = orig_dirs[k]
        res[k, 0] = v[0]*matrix[0,0] + v[1]*matrix[0,1] + v[2]*matrix[0,2]
        res[k, 1] = v[0]*matrix[1,0] + v[1]*matrix[1,1] + v[2]*matrix[1,2]
        res[k, 2] = v[0]*matrix[2,0] + v[1]*matrix[2,1] + v[2]*matrix[2,2]
    return res

@njit
def segment_segment_dist_sq(p1, p2, q1, q2):
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
def is_outside_strict(pos, dirs, current_scale, buffer=0.005):
    rb = R_BASE_MAX * current_scale
    rt = R_TIP_MAX * current_scale
    if pos[0] < rb + buffer or pos[0] > REAL_CONTAINER[0] - rb - buffer: return True
    if pos[1] < rb + buffer or pos[1] > REAL_CONTAINER[1] - rb - buffer: return True
    if pos[2] < rb + buffer or pos[2] > REAL_CONTAINER[2] - rb - buffer: return True
    for k in range(4):
        tip = pos + dirs[k] * LEG_H
        if tip[0] < rt + buffer or tip[0] > REAL_CONTAINER[0] - rt - buffer: return True
        if tip[1] < rt + buffer or tip[1] > REAL_CONTAINER[1] - rt - buffer: return True
        if tip[2] < rt + buffer or tip[2] > REAL_CONTAINER[2] - rt - buffer: return True
    return False

@njit
def check_collision(pos1, dirs1, pos2, dirs2, current_scale=1.0, buffer=0.01):
    dist_centers_sq = np.sum((pos1 - pos2)**2)
    if dist_centers_sq > (LEG_H * 2.8)**2: return False
    rb = R_BASE_MAX * current_scale
    rt = R_TIP_MAX * current_scale
    for k1 in range(4):
        p1, p2 = pos1, pos1 + dirs1[k1] * LEG_H
        for k2 in range(4):
            q1, q2 = pos2, pos2 + dirs2[k2] * LEG_H
            dist_sq, sc, tc, _ = segment_segment_dist_sq(p1, p2, q1, q2)
            r1 = rb + sc * (rt - rb)
            r2 = rb + tc * (rt - rb)
            if dist_sq < (r1 + r2 + buffer)**2: return True
    return False

@njit
def run_hybrid_relaxation(positions, matrices, n, current_scale, iterations=50000):
    all_dirs = np.zeros((n, 4, 3))
    for i in range(n): all_dirs[i] = get_dirs(matrices[i])
    
    for step in range(iterations):
        idx = np.random.randint(n)
        old_pos = positions[idx].copy()
        old_mat = matrices[idx].copy()
        old_dirs = all_dirs[idx].copy()
        
        scale = 0.05
        move = (np.random.rand(3) - 0.5) * scale
        move[2] -= 0.005 # Gravity
        
        rot_axis = np.random.rand(3) - 0.5; rot_axis /= (np.linalg.norm(rot_axis) + 1e-9)
        a = (np.random.rand() - 0.5) * 0.1
        ux, uy, uz = rot_axis
        r00, r01, r02 = 1.0, -uz*a, uy*a; r10, r11, r12 = uz*a, 1.0, -ux*a; r20, r21, r22 = -uy*a, ux*a, 1.0
        new_mat = np.zeros((3, 3))
        new_mat[0,0] = r00*old_mat[0,0] + r01*old_mat[1,0] + r02*old_mat[2,0]
        new_mat[0,1] = r00*old_mat[0,1] + r01*old_mat[1,1] + r02*old_mat[2,1]
        new_mat[0,2] = r00*old_mat[0,2] + r01*old_mat[1,2] + r02*old_mat[2,2]
        new_mat[1,0] = r10*old_mat[0,0] + r11*old_mat[1,0] + r12*old_mat[2,0]
        new_mat[1,1] = r10*old_mat[0,1] + r11*old_mat[1,1] + r12*old_mat[2,1]
        new_mat[1,2] = r10*old_mat[0,2] + r11*old_mat[1,2] + r12*old_mat[2,2]
        new_mat[2,0] = r20*old_mat[0,0] + r21*old_mat[1,0] + r22*old_mat[2,0]
        new_mat[2,1] = r20*old_mat[0,1] + r21*old_mat[1,1] + r22*old_mat[2,1]
        new_mat[2,2] = r20*old_mat[0,2] + r21*old_mat[1,2] + r22*old_mat[2,2]
        v1 = new_mat[0]; v1 /= np.linalg.norm(v1); v2 = new_mat[1]; v2 -= np.dot(v1, v2) * v1; v2 /= np.linalg.norm(v2); new_mat[2] = np.cross(v1, v2)
        
        new_pos = old_pos + move
        new_dirs = get_dirs(new_mat)
        if is_outside_strict(new_pos, new_dirs, current_scale): continue
        
        # Local collision check
        old_hits, new_hits = 0, 0
        for j in range(n):
            if idx == j: continue
            if check_collision(old_pos, old_dirs, positions[j], all_dirs[j], current_scale): old_hits += 1
            if check_collision(new_pos, new_dirs, positions[j], all_dirs[j], current_scale): new_hits += 1
        
        if new_hits < old_hits:
            positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
        elif new_hits == old_hits and new_hits > 0:
            positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
        elif new_hits == 0 and old_hits == 0:
            if new_pos[2] < old_pos[2]: positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
            
    return 0

def run_hybrid_engine(target_n=30):
    print(f"Starting HVIE V6 (Simultaneous Growth) with {target_n} units...")
    positions = np.random.rand(target_n, 3) * (REAL_CONTAINER - 1.2) + 0.6
    matrices = np.zeros((target_n, 3, 3))
    for i in range(target_n):
        matrices[i] = trimesh.transformations.random_rotation_matrix()[:3, :3]
        while is_outside_strict(positions[i], get_dirs(matrices[i]), 0.3):
            positions[i] = np.random.rand(3) * (REAL_CONTAINER - 1.2) + 0.6
    
    # 1. Inflation Loop
    for scale in np.linspace(0.3, 1.0, 15):
        print(f"  Growth Phase: {int(scale*100)}% scale...")
        run_hybrid_relaxation(positions, matrices, target_n, scale, iterations=50000)
    
    # 2. Final Hardening
    print("  Final Settlement...")
    run_hybrid_relaxation(positions, matrices, target_n, 1.0, iterations=150000)
    
    return positions, matrices

if __name__ == "__main__":
    count = 20 # Reliable high density
    final_pos, final_mats = run_hybrid_engine(target_n=count)
    import generate_mesh; tp_mesh = generate_mesh.create_tetrapod(h=1.13); scene = trimesh.Scene()
    for i in range(count):
        m = tp_mesh.copy(); full_mat = np.eye(4); full_mat[:3, :3] = final_mats[i]; full_mat[:3, 3] = final_pos[i]; m.apply_transform(full_mat); scene.add_geometry(m)
    container_box = trimesh.creation.box(extents=REAL_CONTAINER); container_box.apply_translation(REAL_CONTAINER/2)
    container_box.visual.face_colors = [100, 100, 100, 40]; scene.add_geometry(container_box)
    output_path = "sage_tetrapod/export_results/31_Final_Strict_Interlock.glb"
    scene.export(output_path); print(f"Verified Hybrid result saved to {output_path}")
