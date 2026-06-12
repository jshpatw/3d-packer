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
def is_outside_contact(pos, dirs, buffer=0.001):
    # Core check (Radius R_BASE)
    for d in range(3):
        if pos[d] < R_BASE + buffer or pos[d] > REAL_CONTAINER[d] - R_BASE - buffer:
            return True
    # Tips check (Radius R_TIP)
    for k in range(4):
        tip = pos + dirs[k] * LEG_H
        for d in range(3):
            if tip[d] < R_TIP + buffer or tip[d] > REAL_CONTAINER[d] - R_TIP - buffer:
                return True
    return False

@njit
def check_collision_contact(pos1, dirs1, pos2, dirs2, buffer=0.001):
    dist_centers_sq = np.sum((pos1 - pos2)**2)
    if dist_centers_sq > (LEG_H * 2.8)**2: return False
    
    for k1 in range(4):
        p1, p2 = pos1, pos1 + dirs1[k1] * LEG_H
        for k2 in range(4):
            q1, q2 = pos2, pos2 + dirs2[k2] * LEG_H
            dist_sq, sc, tc = segment_segment_dist_sq(p1, p2, q1, q2)
            r1 = R_BASE + sc * (R_TIP - R_BASE)
            r2 = R_BASE + tc * (R_TIP - R_BASE)
            # HARD LIMIT: Any distance less than sum of radii + 1mm epsilon is a collision
            if dist_sq < (r1 + r2 + buffer)**2:
                return True
    return False

@njit
def count_collisions(positions, all_dirs, n):
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if check_collision_contact(positions[i], all_dirs[i], positions[j], all_dirs[j]):
                count += 1
    return count

@njit
def run_contact_relaxation(positions, matrices, n, iterations=500000):
    all_dirs = np.zeros((n, 4, 3))
    for i in range(n): all_dirs[i] = get_dirs(matrices[i])
    current_collisions = count_collisions(positions, all_dirs, n)
    
    for step in range(iterations):
        progress = step / iterations
        idx = np.random.randint(n)
        old_pos, old_mat, old_dirs = positions[idx].copy(), matrices[idx].copy(), all_dirs[idx].copy()
        
        # Settle moves - start aggressive, finish fine
        scale = 0.1 * (1.0 - progress)**2 + 0.001
        move = (np.random.rand(3) - 0.5) * scale
        move[2] -= 0.005 * (1.0 - progress) # Gravity pull
        
        rot_axis = np.random.rand(3) - 0.5; rot_axis /= (np.linalg.norm(rot_axis) + 1e-9)
        a = (np.random.rand() - 0.5) * scale * 3.0
        
        # Matrix update
        ux, uy, uz = rot_axis
        r00, r01, r02 = 1.0, -uz*a, uy*a
        r10, r11, r12 = uz*a, 1.0, -ux*a
        r20, r21, r22 = -uy*a, ux*a, 1.0
        m = old_mat
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
        
        new_pos = old_pos + move
        new_dirs = get_dirs(new_mat)
        
        if is_outside_contact(new_pos, new_dirs): continue
        
        # Local collision change
        old_hits, new_hits = 0, 0
        for j in range(n):
            if idx == j: continue
            if check_collision_contact(old_pos, old_dirs, positions[j], all_dirs[j]): old_hits += 1
            if check_collision_contact(new_pos, new_dirs, positions[j], all_dirs[j]): new_hits += 1
            
        # Acceptance
        if new_hits < old_hits:
            positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
            current_collisions += (new_hits - old_hits)
        elif new_hits == old_hits and new_hits > 0:
            positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
        elif new_hits == 0 and old_hits == 0:
            if new_pos[2] < old_pos[2]: # Settle down if zero collisions
                positions[idx], matrices[idx], all_dirs[idx] = new_pos, new_mat, new_dirs
        
        if current_collisions == 0 and progress > 0.9: break
    return current_collisions

def run_contact_optimizer(start_count=28):
    print(f"Starting Surface-Contact Optimizer (Target: {start_count} units)...")
    positions = np.random.rand(start_count, 3) * (REAL_CONTAINER - 1.2) + 0.6
    matrices = np.zeros((start_count, 3, 3))
    for i in range(start_count):
        matrices[i] = trimesh.transformations.random_rotation_matrix()[:3, :3]
        while is_outside_contact(positions[i], get_dirs(matrices[i])):
            positions[i] = np.random.rand(3) * (REAL_CONTAINER - 1.2) + 0.6

    curr_n = start_count
    while curr_n > 0:
        print(f"\n--- ATTEMPTING {curr_n} UNITS ---")
        # Run MC relaxation with 1mm safety epsilon
        collisions = run_contact_relaxation(positions[:curr_n], matrices[:curr_n], curr_n, iterations=800000)
        print(f"Result: {collisions} penetration pairs.")
        if collisions == 0:
            print(f"SUCCESS: Dense non-penetrating stack found with {curr_n} units!")
            break
        else:
            # Cull the unit with most penetrations
            all_dirs = np.zeros((curr_n, 4, 3))
            for i in range(curr_n): all_dirs[i] = get_dirs(matrices[i])
            hits = np.zeros(curr_n)
            for i in range(curr_n):
                for j in range(i + 1, curr_n):
                    if check_collision_contact(positions[i], all_dirs[i], positions[j], all_dirs[j]):
                        hits[i] += 1; hits[j] += 1
            victim = np.argmax(hits)
            print(f"Culling unit {victim} (had {int(hits[victim])} penetrations)...")
            positions[victim:curr_n-1] = positions[victim+1:curr_n]
            matrices[victim:curr_n-1] = matrices[victim+1:curr_n]
            curr_n -= 1
            
    return positions[:curr_n], matrices[:curr_n], curr_n

if __name__ == "__main__":
    # Start high (28) and let the culling find the true physical limit with contact
    pos, mats, count = run_contact_optimizer(start_count=28)
    import generate_mesh
    tp_mesh = generate_mesh.create_tetrapod(h=1.13); scene = trimesh.Scene()
    for i in range(count):
        m = tp_mesh.copy(); full_mat = np.eye(4); full_mat[:3, :3] = mats[i]; full_mat[:3, 3] = pos[i]; m.apply_transform(full_mat); scene.add_geometry(m)
    container_box = trimesh.creation.box(extents=REAL_CONTAINER); container_box.apply_translation(REAL_CONTAINER/2); container_box.visual.face_colors = [100, 100, 100, 40]; scene.add_geometry(container_box)
    output_path = "sage_tetrapod/export_results/28_SurfaceContact_MaxDensity.glb"
    scene.export(output_path); print(f"Final Density-Optimal Count: {count}. Result saved to {output_path}")
