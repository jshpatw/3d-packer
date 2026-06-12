import trimesh
import numpy as np
import os
import time
from numba import njit, prange

# --- Constants ---
REAL_CONTAINER = np.array([5.898, 2.352, 2.393])
H_UNIT = 1.13
R_BASE = 0.47 * H_UNIT / 2
R_TIP = 0.3 * H_UNIT / 2
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
    return np.dot(diff, diff), sc, tc

@njit
def check_collision_with_packed(pos_new, dirs_new, packed_positions, packed_dirs_list, num_packed, buffer=0.005):
    for i in range(num_packed):
        dist_centers_sq = np.sum((pos_new - packed_positions[i])**2)
        if dist_centers_sq > (LEG_H * 2.8)**2: continue
        for k1 in range(4):
            p1, p2 = pos_new, pos_new + dirs_new[k1] * LEG_H
            for k2 in range(4):
                q1, q2 = packed_positions[i], packed_positions[i] + packed_dirs_list[i, k2] * LEG_H
                dist_sq, sc, tc = segment_segment_dist_sq(p1, p2, q1, q2)
                r1 = R_BASE + sc * (R_TIP - R_BASE)
                r2 = R_BASE + tc * (R_TIP - R_BASE)
                if dist_sq < (r1 + r2 + buffer)**2: return True
    return False

@njit
def is_outside(pos, dirs, buffer=0.01):
    if pos[0] < 0.1 or pos[0] > REAL_CONTAINER[0]-0.1: return True
    if pos[1] < 0.1 or pos[1] > REAL_CONTAINER[1]-0.1: return True
    if pos[2] < 0.1 or pos[2] > REAL_CONTAINER[2]-0.1: return True
    for k in range(4):
        tip = pos + dirs[k] * LEG_H
        if tip[0] < buffer or tip[0] > REAL_CONTAINER[0]-buffer: return True
        if tip[1] < buffer or tip[1] > REAL_CONTAINER[1]-buffer: return True
        if tip[2] < buffer or tip[2] > REAL_CONTAINER[2]-buffer: return True
    return False

@njit
def mc_settle_interlocking(start_pos, start_mat, packed_positions, packed_dirs_list, num_packed, iterations=40000):
    curr_pos = start_pos.copy()
    curr_mat = start_mat.copy()
    curr_dirs = get_dirs(curr_mat)
    best_z = 1e9
    best_pos = curr_pos.copy()
    best_mat = curr_mat.copy()
    found_valid = False
    if num_packed == 0: found_valid = True

    for i in range(iterations):
        progress = i / iterations
        scale = 0.12 * (1.0 - progress) + 0.002
        move = (np.random.rand(3) - 0.5) * scale
        move[2] -= 0.02 # Gravity
        
        rot_axis = np.random.rand(3) - 0.5
        rot_axis /= (np.linalg.norm(rot_axis) + 1e-9)
        a = (np.random.rand() - 0.5) * scale * 2.5
        
        ux, uy, uz = rot_axis
        r00, r01, r02 = 1.0, -uz*a, uy*a
        r10, r11, r12 = uz*a, 1.0, -ux*a
        r20, r21, r22 = -uy*a, ux*a, 1.0
        
        m = curr_mat
        new_mat = np.zeros((3, 3))
        new_mat[0,0] = r00*m[0,0] + r01*m[1,0] + r02*m[2,0]
        new_mat[0,1] = r00*m[0,1] + r01*m[1,1] + r02*m[2,1]
        new_mat[0,2] = r00*m[0,2] + r01*m[1,2] + r02*m[2,2]
        new_mat[1,0] = r10*m[0,0] + r11*m[1,0] + r12*m[2,0]
        new_mat[1,1] = r10*m[0,1] + r11*m[1,1] + r12*m[2,1]
        new_mat[1,2] = r10*m[0,2] + r11*m[1,2] + r12*m[2,2]
        new_mat[2,0] = r20*m[0,0] + r21*m[1,0] + r22*m[2,0]
        new_mat[2,1] = r20*m[0,1] + r21*m[1,1] + r22*m[2,1]
        new_mat[2,2] = r20*m[0,2] + r21*m[1,2] + r22*m[2,2]
        
        v1 = new_mat[0]; v1 /= np.linalg.norm(v1)
        v2 = new_mat[1]; v2 -= np.dot(v1, v2) * v1; v2 /= np.linalg.norm(v2)
        new_mat[2] = np.cross(v1, v2)
        
        new_pos = curr_pos + move
        new_dirs = get_dirs(new_mat)
        if is_outside(new_pos, new_dirs): continue
        
        collided = check_collision_with_packed(new_pos, new_dirs, packed_positions, packed_dirs_list, num_packed)
        
        if progress < 0.35: # Ghost phase
            curr_pos, curr_mat, curr_dirs = new_pos, new_mat, new_dirs
            if not collided:
                found_valid = True
                if curr_pos[2] < best_z:
                    best_z, best_pos, best_mat = curr_pos[2], curr_pos.copy(), curr_mat.copy()
        else:
            if not collided:
                found_valid = True
                if new_pos[2] <= curr_pos[2] + 0.005:
                    curr_pos, curr_mat, curr_dirs = new_pos, new_mat, new_dirs
                    if curr_pos[2] < best_z:
                        best_z, best_pos, best_mat = curr_pos[2], curr_pos.copy(), curr_mat.copy()
    return best_pos, best_mat, best_z

@njit(parallel=True)
def coarse_search(seeds_pos, seeds_mat, packed_positions, packed_dirs_list, num_packed):
    n = seeds_pos.ssage[0]
    results_pos = np.zeros((n, 3))
    results_mat = np.zeros((n, 3, 3))
    results_z = np.zeros(n)
    for i in prange(n):
        p, m, z = mc_settle_interlocking(seeds_pos[i], seeds_mat[i], packed_positions, packed_dirs_list, num_packed, iterations=8000)
        results_pos[i] = p
        results_mat[i] = m
        results_z[i] = z
    return results_pos, results_mat, results_z

@njit(parallel=True)
def refined_search(seeds_pos, seeds_mat, packed_positions, packed_dirs_list, num_packed):
    n = seeds_pos.ssage[0]
    results_pos = np.zeros((n, 3))
    results_mat = np.zeros((n, 3, 3))
    results_z = np.zeros(n)
    for i in prange(n):
        # INCREASED ITERATIONS to 150k
        p, m, z = mc_settle_interlocking(seeds_pos[i], seeds_mat[i], packed_positions, packed_dirs_list, num_packed, iterations=150000)
        results_pos[i] = p
        results_mat[i] = m
        results_z[i] = z
    return results_pos, results_mat, results_z

def run_hierarchical_stacker(target_count=60):
    print(f"Starting SAGE Stacker V23 (Spatial Hierarchical)...")
    packed_positions = np.zeros((target_count, 3))
    packed_matrices = np.zeros((target_count, 3, 3))
    packed_dirs_list = np.zeros((target_count, 4, 3))
    num_packed = 0
    
    start_time = time.time()
    for i in range(target_count):
        print(f"Packing Unit {i+1}...", end=" ", flush=True)
        
        # 1. Coarse Search: 2048 seeds
        num_coarse = 2048
        coarse_seeds_pos = np.random.rand(num_coarse, 3)
        coarse_seeds_pos[:, 0] = coarse_seeds_pos[:, 0] * (REAL_CONTAINER[0]-1.2) + 0.6
        coarse_seeds_pos[:, 1] = coarse_seeds_pos[:, 1] * (REAL_CONTAINER[1]-1.2) + 0.6
        coarse_seeds_pos[:, 2] = coarse_seeds_pos[:, 2] * (REAL_CONTAINER[2]-1.0) + 0.5
        
        coarse_seeds_mat = np.zeros((num_coarse, 3, 3))
        for t in range(num_coarse):
            if t < 100: coarse_seeds_mat[t] = np.eye(3)
            elif t < 200: coarse_seeds_mat[t] = trimesh.transformations.euler_matrix(np.arccos(-1/3), 0, 0)[:3, :3]
            else: coarse_seeds_mat[t] = trimesh.transformations.random_rotation_matrix()[:3, :3]
        
        c_pos, c_mat, c_z = coarse_search(coarse_seeds_pos, coarse_seeds_mat, packed_positions, packed_dirs_list, num_packed)
        
        # SPATIAL SEEDING: Pick best from each grid cell
        best_seeds_indices = []
        x_divs = np.linspace(0.6, REAL_CONTAINER[0]-0.6, 7)
        y_divs = np.linspace(0.6, REAL_CONTAINER[1]-0.6, 4)
        
        for xi in range(6):
            for yi in range(3):
                mask = (c_pos[:, 0] >= x_divs[xi]) & (c_pos[:, 0] < x_divs[xi+1]) & \
                       (c_pos[:, 1] >= y_divs[yi]) & (c_pos[:, 1] < y_divs[yi+1]) & \
                       (c_z < 1e5)
                if np.any(mask):
                    cell_indices = np.where(mask)[0]
                    best_seeds_indices.append(cell_indices[np.argmin(c_z[mask])])
        
        if not best_seeds_indices:
            print("Failed (No valid coarse seeds).")
            break
            
        top_indices = np.array(best_seeds_indices)
        
        # 2. Refined Search: 150,000 iterations
        r_pos, r_mat, r_z = refined_search(c_pos[top_indices], c_mat[top_indices], packed_positions, packed_dirs_list, num_packed)
        best_idx = np.argmin(r_z)
        
        if r_z[best_idx] < 2.35:
            packed_positions[num_packed] = r_pos[best_idx]
            packed_matrices[num_packed] = r_mat[best_idx]
            packed_dirs_list[num_packed] = get_dirs(r_mat[best_idx])
            num_packed += 1
            print(f"Success! (z={r_z[best_idx]:.2f})")
        else:
            print("Failed (Refined search jammed).")
            break
            
    print(f"Total: {num_packed} units in {time.time()-start_time:.2f}s")
    return packed_positions[:num_packed], packed_matrices[:num_packed], num_packed

if __name__ == "__main__":
    pos, mats, count = run_hierarchical_stacker()
    import generate_mesh
    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
    scene = trimesh.Scene()
    for i in range(count):
        m = tp_mesh.copy()
        full_mat = np.eye(4); full_mat[:3, :3] = mats[i]; full_mat[:3, 3] = pos[i]
        m.apply_transform(full_mat); scene.add_geometry(m)
    container_box = trimesh.creation.box(extents=REAL_CONTAINER)
    container_box.apply_translation(REAL_CONTAINER/2)
    container_box.visual.face_colors = [100, 100, 100, 40]
    scene.add_geometry(container_box)
    scene.export("sage_tetrapod/export_results/25_HierarchicalStacker_HighDensity.glb")
