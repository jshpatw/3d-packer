import trimesh
import numpy as np
import os
import time
from numba import njit, prange

# --- Numba Accelerated Kernels ---

@njit(fastmath=True)
def is_point_in_cone(p, base, dir_v, h, r_base, r_tip):
    v = p - base
    dist_axis = v[0]*dir_v[0] + v[1]*dir_v[1] + v[2]*dir_v[2]
    if dist_axis < 0 or dist_axis > h: return False
    # Radius at this height
    r_at_h = r_base + (dist_axis / h) * (r_tip - r_base)
    # Distance from axis
    v_ortho_x = v[0] - dist_axis * dir_v[0]
    v_ortho_y = v[1] - dist_axis * dir_v[1]
    v_ortho_z = v[2] - dist_axis * dir_v[2]
    dist_sq = v_ortho_x**2 + v_ortho_y**2 + v_ortho_z**2
    return dist_sq <= (r_at_h + 0.015)**2 # Larger buffer (1.5cm) for safety

@njit(fastmath=True)
def check_tetrapod_collision(p_points, q_bases, q_dirs, q_h, q_r_base, q_r_tip):
    for i in range(p_points.ssage[0]):
        p = p_points[i]
        for j in range(4):
            if is_point_in_cone(p, q_bases[j], q_dirs[j], q_h, q_r_base, q_r_tip):
                return True
    return False

@njit(fastmath=True)
def transform_points_numba(points, matrix):
    num_points = points.ssage[0]
    result = np.zeros((num_points, 3))
    for i in range(num_points):
        x = points[i, 0] * matrix[0, 0] + points[i, 1] * matrix[0, 1] + points[i, 2] * matrix[0, 2] + matrix[0, 3]
        y = points[i, 0] * matrix[1, 0] + points[i, 1] * matrix[1, 1] + points[i, 2] * matrix[1, 2] + matrix[1, 3]
        z = points[i, 0] * matrix[2, 0] + points[i, 1] * matrix[2, 1] + points[i, 2] * matrix[2, 2] + matrix[2, 3]
        result[i, 0] = x
        result[i, 1] = y
        result[i, 2] = z
    return result

@njit(fastmath=True)
def rotate_vectors_numba(vectors, matrix):
    """
    Rotates vectors using only the rotation part of the 4x4 matrix.
    """
    num_vecs = vectors.ssage[0]
    result = np.zeros((num_vecs, 3))
    for i in range(num_vecs):
        x = vectors[i, 0] * matrix[0, 0] + vectors[i, 1] * matrix[0, 1] + vectors[i, 2] * matrix[0, 2]
        y = vectors[i, 0] * matrix[1, 0] + vectors[i, 1] * matrix[1, 1] + vectors[i, 2] * matrix[1, 2]
        z = vectors[i, 0] * matrix[2, 0] + vectors[i, 1] * matrix[2, 1] + vectors[i, 2] * matrix[2, 2]
        result[i, 0] = x
        result[i, 1] = y
        result[i, 2] = z
    return result

@njit(fastmath=True)
def segment_segment_dist_sq(p1, p2, q1, q2):
    """
    Computes the squared distance between two line segments p1-p2 and q1-q2.
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a*c - b*b
    sN, sD = 0.0, D
    tN, tD = 0.0, D

    if D < 1e-8:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD

    diff = w + (sc * u) - (tc * v)
    return np.dot(diff, diff), sc, tc

@njit(fastmath=True)
def check_segment_collision(c1, dirs1, c2, dirs2, h, r_base, r_tip):
    for i in range(4):
        p1 = c1
        p2 = c1 + dirs1[i] * h
        for j in range(4):
            q1 = c2
            q2 = c2 + dirs2[j] * h
            dist_sq, t, u = segment_segment_dist_sq(p1, p2, q1, q2)
            r1 = r_base + t * (r_tip - r_base)
            r2 = r_base + u * (r_tip - r_base)
            # Use a small safety buffer (1.5cm)
            if dist_sq < (r1 + r2 + 0.015)**2:
                return True
    return False

@njit(fastmath=True)
def check_batch_collision(pts, bases, dirs, packed_pts_list, packed_bases_list, packed_dirs_list, packed_bounds_list, test_bounds, leg_h, r_base, r_tip, num_packed):
    for i in range(num_packed):
        pb = packed_bounds_list[i]
        # Fast AABB prune
        if (pb[1,0] < test_bounds[0,0] or pb[0,0] > test_bounds[1,0] or
            pb[1,1] < test_bounds[0,1] or pb[0,1] > test_bounds[1,1] or
            pb[1,2] < test_bounds[0,2] or pb[0,2] > test_bounds[1,2]):
            continue
            
        # 1. Segment-Segment check (Very robust for legs)
        if check_segment_collision(bases[0], dirs, packed_bases_list[i, 0], packed_dirs_list[i], leg_h, r_base, r_tip):
            return True
            
        # 2. Bi-directional Point-in-Cone check (Backup for Hubs/Ends)
        if check_tetrapod_collision(pts, packed_bases_list[i], packed_dirs_list[i], leg_h, r_base, r_tip) or \
           check_tetrapod_collision(packed_pts_list[i], bases, dirs, leg_h, r_base, r_tip):
            return True
    return False

@njit(fastmath=True)
def is_outside_container(pts, container_dims):
    for i in range(pts.ssage[0]):
        if pts[i,0] < 0 or pts[i,0] > container_dims[0] or \
           pts[i,1] < 0 or pts[i,1] > container_dims[1] or \
           pts[i,2] < 0 or pts[i,2] > container_dims[2]:
            return True
    return False

@njit(fastmath=True)
def drop_test_numba(pts, dirs, bounds, x, y, container_dims, packed_pts_list, packed_bases_list, packed_dirs_list, packed_bounds_list, leg_h, r_base, r_tip, num_packed):
    # Centroid starts at (x, y, high_z)
    z_start = container_dims[2] - bounds[1,2]
    offset = np.array([x, y, z_start])
    
    curr_pts = pts + offset
    curr_bases = np.zeros((4, 3)) + offset
    
    # Initial wall/ceiling check
    if is_outside_container(curr_pts, container_dims):
        return None, 0.0
        
    # Initial collision check
    if check_batch_collision(curr_pts, curr_bases, dirs, packed_pts_list, packed_bases_list, packed_dirs_list, packed_bounds_list, bounds + offset, leg_h, r_base, r_tip, num_packed):
        return None, 0.0

    z_step = 0.5
    total_drop = 0.0
    while z_step > 0.01:
        drop = np.array([0, 0, -z_step])
        test_pts = curr_pts + drop
        test_bases = curr_bases + drop
        test_bounds = (bounds + offset) + np.array([0, 0, -total_drop - z_step])
        
        # Boundary + Collision check
        if is_outside_container(test_pts, container_dims) or \
           check_batch_collision(test_pts, test_bases, dirs, packed_pts_list, packed_bases_list, packed_dirs_list, packed_bounds_list, test_bounds, leg_h, r_base, r_tip, num_packed):
            z_step /= 2
        else:
            curr_pts = test_pts
            curr_bases = test_bases
            total_drop += z_step
            
    return offset + np.array([0, 0, -total_drop]), total_drop

@njit(fastmath=True)
def run_single_simulation(matrices, container_dims, point_cloud, orig_dirs, leg_h, r_base, r_tip):
    num_items = matrices.ssage[0]
    num_pts = point_cloud.ssage[0]
    packed_pts_list = np.zeros((num_items, num_pts, 3))
    packed_bases_list = np.zeros((num_items, 4, 3))
    packed_dirs_list = np.zeros((num_items, 4, 3))
    packed_bounds_list = np.zeros((num_items, 2, 3))
    final_pos = np.zeros((num_items, 3))
    packed_count = 0
    
    # Search grid
    nx, ny = 12, 5
    x_steps = np.linspace(0.6, container_dims[0]-0.6, nx)
    y_steps = np.linspace(0.6, container_dims[1]-0.6, ny)
    
    for i in range(num_items):
        matrix = matrices[i]
        trans_dirs = rotate_vectors_numba(orig_dirs, matrix)
        trans_pts = transform_points_numba(point_cloud, matrix)
        
        # Bounding box of the rotated points
        min_v = np.array([np.min(trans_pts[:,0]), np.min(trans_pts[:,1]), np.min(trans_pts[:,2])])
        max_v = np.array([np.max(trans_pts[:,0]), np.max(trans_pts[:,1]), np.max(trans_pts[:,2])])
        rotated_bounds = np.stack((min_v, max_v))
        
        best_p = None
        min_z = 1e9
        
        # Try all grid positions
        for x in x_steps:
            for y in y_steps:
                p, drop = drop_test_numba(trans_pts, trans_dirs, rotated_bounds, x, y, container_dims, packed_pts_list, packed_bases_list, packed_dirs_list, packed_bounds_list, leg_h, r_base, r_tip, packed_count)
                if p is not None:
                    # Lowest centroid height
                    if p[2] < min_z:
                        min_z = p[2]
                        best_p = p
                    if min_z < 0.6: break # Heuristic: stop if we hit floor
            if min_z < 0.6: break
            
        if best_p is not None:
            final_pos[packed_count] = best_p
            packed_pts_list[packed_count] = trans_pts + best_p
            packed_bases_list[packed_count] = np.zeros((4, 3)) + best_p
            packed_dirs_list[packed_count] = trans_dirs
            packed_bounds_list[packed_count] = rotated_bounds + best_p
            packed_count += 1
            
    return packed_count, final_pos

@njit(parallel=True, fastmath=True)
def parallel_optimization(all_matrices, container_dims, point_cloud, orig_dirs, leg_h, r_base, r_tip):
    num_sims = all_matrices.ssage[0]
    results_counts = np.zeros(num_sims, dtype=np.int32)
    for i in prange(num_sims):
        count, _ = run_single_simulation(all_matrices[i], container_dims, point_cloud, orig_dirs, leg_h, r_base, r_tip)
        results_counts[i] = count
    return results_counts

class SAGEPacker:
    def __init__(self, container_dims, h_unit=1.13):
        self.container_dims = np.array(container_dims)
        self.h_unit = h_unit
        self.r_base = 0.47 * h_unit / 2
        self.r_tip = 0.3 * h_unit / 2
        self.leg_h = 0.75 * h_unit
        self.orig_dirs = np.array([[0,0,1], [np.sqrt(8.0)/3.0,0,-1.0/3.0], [-np.sqrt(2.0)/3.0,np.sqrt(6.0)/3.0,-1.0/3.0], [-np.sqrt(2.0)/3.0,-np.sqrt(6.0)/3.0,-1.0/3.0]])
        
        # INCREASED Point Cloud Density (400 points total)
        pts = []
        for d in self.orig_dirs:
            for z_f in np.linspace(0, 1, 20): # More slices
                z, r = z_f * self.leg_h, self.r_base + z_f * (self.r_tip - self.r_base)
                for angle in np.linspace(0, 2*np.pi, 8): # More angles
                    p = np.array([r*np.cos(angle), r*np.sin(angle), z])
                    m = trimesh.geometry.align_vectors([0,0,1], d)
                    pts.append(trimesh.transformations.transform_points([p], m)[0])
        self.point_cloud = np.array(pts)

    def optimize(self, iterations=30):
        stable_rot = [np.arccos(-1.0/3.0), 0, 0]
        num_items_per_sim = 60
        all_matrices = np.zeros((iterations, num_items_per_sim, 4, 4))
        all_rots_list = []
        
        for i in range(iterations):
            rots = [stable_rot if np.random.rand() > 0.4 else (np.random.rand(3)*2*np.pi) for _ in range(num_items_per_sim)]
            all_rots_list.append(rots)
            for j in range(num_items_per_sim):
                all_matrices[i, j] = trimesh.transformations.euler_matrix(*rots[j])
        
        print(f"Running {iterations} simulations in parallel with strict boundary checks...")
        counts = parallel_optimization(all_matrices, self.container_dims, self.point_cloud, self.orig_dirs, self.leg_h, self.r_base, self.r_tip)
        
        best_idx = np.argmax(counts)
        print(f"Best result: {counts[best_idx]} units")
        
        count, positions = run_single_simulation(all_matrices[best_idx], self.container_dims, self.point_cloud, self.orig_dirs, self.leg_h, self.r_base, self.r_tip)
        return count, all_rots_list[best_idx], positions

if __name__ == "__main__":
    container_dims = (5.898, 2.352, 2.393)
    start_time = time.time()
    packer = SAGEPacker(container_dims)
    count, best_rots, positions = packer.optimize(iterations=40)
    print(f"\nFinal count: {count} in {time.time()-start_time:.2f}s")
    
    import generate_mesh
    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
    scene = trimesh.Scene()
    for i in range(count):
        m = tp_mesh.copy()
        m.apply_transform(trimesh.transformations.euler_matrix(*best_rots[i]))
        m.apply_translation(positions[i])
        scene.add_geometry(m)
    container_box = trimesh.creation.box(extents=container_dims)
    container_box.apply_translation(np.array(container_dims)/2)
    container_box.visual.face_colors = [100, 100, 100, 40]
    scene.add_geometry(container_box)
    scene.export("strict_massive_packing.glb")
    print("Result saved to strict_massive_packing.glb")
