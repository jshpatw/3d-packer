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
def check_single_collision(pos_i, dirs_i, pos_j, dirs_j, buffer=0.03):
    for k1 in range(4):
        p1, p2 = pos_i, pos_i + dirs_i[k1] * LEG_H
        for k2 in range(4):
            q1, q2 = pos_j, pos_j + dirs_j[k2] * LEG_H
            dist_sq, sc, tc = segment_segment_dist_sq(p1, p2, q1, q2)
            r_i = R_BASE + sc * (R_TIP - R_BASE)
            r_j = R_BASE + tc * (R_TIP - R_BASE)
            if dist_sq < (r_i + r_j + buffer)**2:
                return True
    return False

@njit
def is_outside(pos, dirs):
    for d in range(3):
        if pos[d] < R_BASE + 0.01 or pos[d] > REAL_CONTAINER[d] - R_BASE - 0.01:
            return True
    for k in range(4):
        tip = pos + dirs[k] * LEG_H
        for d in range(3):
            if tip[d] < R_TIP + 0.01 or tip[d] > REAL_CONTAINER[d] - R_TIP - 0.01:
                return True
    return False

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

def run_mc_guaranteed(num_items=18, iterations_per_level=40000):
    print(f"Starting Guaranteed Monte Carlo ({num_items} units initial)...")
    
    # Initialize
    positions = np.random.rand(num_items, 3) * (REAL_CONTAINER - 1.2) + 0.6
    matrices = np.zeros((num_items, 3, 3))
    all_dirs = np.zeros((num_items, 4, 3))
    for i in range(num_items):
        matrices[i] = trimesh.transformations.random_rotation_matrix()[:3, :3]
        all_dirs[i] = get_dirs(matrices[i])
        while is_outside(positions[i], all_dirs[i]):
            positions[i] = np.random.rand(3) * (REAL_CONTAINER - 1.2) + 0.6

    active_indices = list(range(num_items))
    
    while len(active_indices) > 0:
        n = len(active_indices)
        print(f"\n--- ATTEMPTING LEVEL: {n} UNITS ---")
        
        # Calculate current collisions
        collision_matrix = np.zeros((n, n), dtype=bool)
        total_collisions = 0
        for i_idx, i in enumerate(active_indices):
            for j_idx, j in enumerate(active_indices):
                if i_idx >= j_idx: continue
                if check_single_collision(positions[i], all_dirs[i], positions[j], all_dirs[j]):
                    collision_matrix[i_idx, j_idx] = True
                    total_collisions += 1
        
        print(f"Starting collisions: {total_collisions}")
        if total_collisions == 0:
            print("Target reached!")
            break

        # Monte Carlo Loop
        for step in range(iterations_per_level):
            # Pick random active unit
            local_idx = np.random.randint(n)
            idx = active_indices[local_idx]
            
            old_pos = positions[idx].copy()
            old_mat = matrices[idx].copy()
            old_dirs = all_dirs[idx].copy()
            
            scale = 0.1 * (1.0 - step/iterations_per_level) + 0.002
            move = (np.random.rand(3) - 0.5) * scale
            rand_mat = trimesh.transformations.rotation_matrix((np.random.rand()-0.5)*scale*2, np.random.rand(3))[:3, :3]
            
            new_pos = old_pos + move
            new_mat = rand_mat @ old_mat
            new_dirs = get_dirs(new_mat)
            
            if is_outside(new_pos, new_dirs): continue
            
            # Local collision check
            old_hits = 0
            new_hits = 0
            for other_local_idx, other_idx in enumerate(active_indices):
                if other_idx == idx: continue
                if check_single_collision(old_pos, old_dirs, positions[other_idx], all_dirs[other_idx]):
                    old_hits += 1
                if check_single_collision(new_pos, new_dirs, positions[other_idx], all_dirs[other_idx]):
                    new_hits += 1
            
            # Acceptance
            if new_hits < old_hits:
                positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
                total_collisions += (new_hits - old_hits)
            elif new_hits == old_hits and new_hits > 0:
                positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
            elif new_hits == 0 and old_hits == 0:
                if new_pos[2] < old_pos[2]: # Settle down
                    positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs

            if total_collisions == 0:
                print(f"SUCCESS: Zero collisions reached with {n} units at step {step}")
                break
            
            if step % 10000 == 0 and step > 0:
                print(f"  Step {step:5d} | Collisions: {total_collisions:3d}")

        if total_collisions == 0:
            break
        else:
            # FAILED to reach zero. Cull the unit with most collisions.
            print(f"FAILED to reach zero. Culling unit...")
            collision_counts = np.zeros(n)
            for i_idx in range(n):
                for j_idx in range(n):
                    if i_idx == j_idx: continue
                    a, b = min(i_idx, j_idx), max(i_idx, j_idx)
                    if check_single_collision(positions[active_indices[i_idx]], all_dirs[active_indices[i_idx]], 
                                            positions[active_indices[j_idx]], all_dirs[active_indices[j_idx]]):
                        collision_counts[i_idx] += 1
            
            victim_local = np.argmax(collision_counts)
            print(f"Removing unit {active_indices[victim_local]} (had {int(collision_counts[victim_local])} collisions)")
            active_indices.pop(victim_local)
            # Loop continues with n-1 units

    # Final positions
    final_pos = [positions[i] for i in active_indices]
    final_mats = [matrices[i] for i in active_indices]
    return final_pos, final_mats, len(active_indices)

if __name__ == "__main__":
    try:
        pos, mats, count = run_mc_guaranteed(num_items=18, iterations_per_level=60000)
        
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            m = tp_mesh.copy()
            full_mat = np.eye(4)
            full_mat[:3, :3] = mats[i]
            full_mat[:3, 3] = pos[i]
            m.apply_transform(full_mat)
            scene.add_geometry(m)
            
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 40]
        scene.add_geometry(container_box)
        
        output_path = "sage_tetrapod/export_results/23_MonteCarlo_ZeroCollision.glb"
        scene.export(output_path)
        print(f"Zero-Collision result saved to {output_path} (Final Count: {count})")
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
