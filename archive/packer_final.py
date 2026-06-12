import trimesh
import numpy as np
import os
import time
from numba import njit, prange

# --- Constants ---
REAL_CONTAINER = np.array([5.898, 2.352, 2.393])

# --- Numba Accelerated Kernels ---

@njit(fastmath=True)
def segment_segment_dist_sq(p1, p2, q1, q2):
    u, v, w = p2 - p1, q2 - q1, p1 - q1
    a, b, c, d, e = np.dot(u, u), np.dot(u, v), np.dot(v, v), np.dot(u, w), np.dot(v, w)
    D = a*c - b*b
    sN, sD = 0.0, D
    tN, tD = 0.0, D
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
    diff = w + (sc * u) - (tc * v)
    return np.dot(diff, diff), sc, tc

@njit(fastmath=True)
def check_segment_collision(c1, dirs1, c2, dirs2, h, r_base, r_tip):
    for i in range(4):
        p1, p2 = c1, c1 + dirs1[i] * h
        for j in range(4):
            q1, q2 = c2, c2 + dirs2[j] * h
            dist_sq, t, u = segment_segment_dist_sq(p1, p2, q1, q2)
            r1, r2 = r_base + t * (r_tip - r_base), r_base + u * (r_tip - r_base)
            if dist_sq < (r1 + r2 + 0.02)**2: return True
    return False

@njit(fastmath=True)
def is_outside_container_strict(pos, dirs, h, r_tip, container_dims):
    # Check all 4 leg tips against walls
    for i in range(4):
        tip = pos + dirs[i] * h
        # We must account for the radius of the tip!
        if tip[0] < r_tip or tip[0] > container_dims[0] - r_tip or \
           tip[1] < r_tip or tip[1] > container_dims[1] - r_tip or \
           tip[2] < r_tip or tip[2] > container_dims[2] - r_tip:
            return True
    # Check central hub (radius is r_base)
    if pos[0] < 0.23 or pos[0] > container_dims[0] - 0.23 or \
       pos[1] < 0.23 or pos[1] > container_dims[1] - 0.23 or \
       pos[2] < 0.23 or pos[2] > container_dims[2] - 0.23:
        return True
    return False

@njit(fastmath=True)
def check_batch_collision(pos, dirs, packed_pos_list, packed_dirs_list, leg_h, r_base, r_tip, num_packed):
    for i in range(num_packed):
        # AABB check for centroids
        dist_sq = np.sum((pos - packed_pos_list[i])**2)
        if dist_sq > (leg_h * 2.5)**2: continue
        
        if check_segment_collision(pos, dirs, packed_pos_list[i], packed_dirs_list[i], leg_h, r_base, r_tip):
            return True
    return False

@njit(fastmath=True)
def drop_test_numba(dirs, x, y, container_dims, packed_pos_list, packed_dirs_list, leg_h, r_base, r_tip, num_packed):
    z_start = container_dims[2] - 0.6
    pos = np.array([x, y, z_start])
    
    if is_outside_container_strict(pos, dirs, leg_h, r_tip, container_dims) or \
       check_batch_collision(pos, dirs, packed_pos_list, packed_dirs_list, leg_h, r_base, r_tip, num_packed):
        return None

    z_step = 0.5
    while z_step > 0.005:
        test_pos = pos + np.array([0, 0, -z_step])
        if is_outside_container_strict(test_pos, dirs, leg_h, r_tip, container_dims) or \
           check_batch_collision(test_pos, dirs, packed_pos_list, packed_dirs_list, leg_h, r_base, r_tip, num_packed):
            z_step /= 2
        else:
            pos = test_pos
            
    return pos

@njit(fastmath=True)
def run_single_simulation(matrices, container_dims, orig_dirs, leg_h, r_base, r_tip):
    num_items = matrices.ssage[0]
    packed_pos_list = np.zeros((num_items, 3))
    packed_dirs_list = np.zeros((num_items, 4, 3))
    packed_count = 0
    
    nx, ny = 15, 6
    x_steps = np.linspace(0.6, container_dims[0]-0.6, nx)
    y_steps = np.linspace(0.6, container_dims[1]-0.6, ny)
    
    for i in range(num_items):
        matrix = matrices[i]
        # Rotate directions
        dirs = np.zeros((4, 3))
        for j in range(4):
            v = orig_dirs[j]
            dirs[j, 0] = v[0]*matrix[0,0] + v[1]*matrix[0,1] + v[2]*matrix[0,2]
            dirs[j, 1] = v[0]*matrix[1,0] + v[1]*matrix[1,1] + v[2]*matrix[1,2]
            dirs[j, 2] = v[0]*matrix[2,0] + v[1]*matrix[2,1] + v[2]*matrix[2,2]
            
        best_p = None
        min_z = 1e9
        for x in x_steps:
            for y in y_steps:
                p = drop_test_numba(dirs, x, y, container_dims, packed_pos_list, packed_dirs_list, leg_h, r_base, r_tip, packed_count)
                if p is not None:
                    if p[2] < min_z:
                        min_z, best_p = p[2], p
                    if min_z < 0.6: break
            if min_z < 0.6: break
            
        if best_p is not None:
            packed_pos_list[packed_count] = best_p
            packed_dirs_list[packed_count] = dirs
            packed_count += 1
            
    return packed_count, packed_pos_list

@njit(parallel=True, fastmath=True)
def parallel_optimization(all_matrices, container_dims, orig_dirs, leg_h, r_base, r_tip):
    num_sims = all_matrices.ssage[0]
    results_counts = np.zeros(num_sims, dtype=np.int32)
    for i in prange(num_sims):
        count, _ = run_single_simulation(all_matrices[i], container_dims, orig_dirs, leg_h, r_base, r_tip)
        results_counts[i] = count
    return results_counts

if __name__ == "__main__":
    h_unit = 1.13
    r_base = 0.47 * h_unit / 2
    r_tip = 0.3 * h_unit / 2
    leg_h = 0.75 * h_unit
    orig_dirs = np.array([[0,0,1], [np.sqrt(8.0)/3.0,0,-1.0/3.0], [-np.sqrt(2.0)/3.0,np.sqrt(6.0)/3.0,-1.0/3.0], [-np.sqrt(2.0)/3.0,-np.sqrt(6.0)/3.0,-1.0/3.0]])

    num_sims = 100
    num_items = 60
    stable_rot = [np.arccos(-1.0/3.0), 0, 0]
    
    all_matrices = np.zeros((num_sims, num_items, 4, 4))
    all_rots = []
    
    for i in range(num_sims):
        rots = [stable_rot if np.random.rand() > 0.4 else (np.random.rand(3)*2*np.pi) for _ in range(num_items)]
        all_rots.append(rots)
        for j in range(num_items):
            all_matrices[i, j] = trimesh.transformations.euler_matrix(*rots[j])
            
    print(f"Running {num_sims} simulations with ANALYTICAL boundary logic...")
    counts = parallel_optimization(all_matrices, REAL_CONTAINER, orig_dirs, leg_h, r_base, r_tip)
    best_idx = np.argmax(counts)
    print(f"Best result: {counts[best_idx]} units")
    
    count, final_pos = run_single_simulation(all_matrices[best_idx], REAL_CONTAINER, orig_dirs, leg_h, r_base, r_tip)
    
    import generate_mesh
    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
    scene = trimesh.Scene()
    for i in range(count):
        m = tp_mesh.copy()
        m.apply_transform(all_matrices[best_idx, i])
        m.apply_translation(final_pos[i])
        scene.add_geometry(m)
        
    container_box = trimesh.creation.box(extents=REAL_CONTAINER)
    container_box.apply_translation(REAL_CONTAINER/2)
    container_box.visual.face_colors = [100, 100, 100, 40]
    scene.add_geometry(container_box)
    
    output_path = "sage_tetrapod/export_results/13_FinalCombinedPacker_Analytical.glb"
    scene.export(output_path)
    print(f"Result saved to {output_path}")
