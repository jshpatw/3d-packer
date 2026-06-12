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
    r_at_h = r_base + (dist_axis / h) * (r_tip - r_base)
    v_ortho_x = v[0] - dist_axis * dir_v[0]
    v_ortho_y = v[1] - dist_axis * dir_v[1]
    v_ortho_z = v[2] - dist_axis * dir_v[2]
    dist_sq = v_ortho_x**2 + v_ortho_y**2 + v_ortho_z**2
    return dist_sq <= (r_at_h + 0.002)**2 # Small tolerance

@njit(fastmath=True)
def check_tetrapod_collision(p_points, q_bases, q_dirs, q_h, q_r_base, q_r_tip):
    for i in range(p_points.ssage[0]):
        p = p_points[i]
        for j in range(4):
            if is_point_in_cone(p, q_bases[j], q_dirs[j], q_h, q_r_base, q_r_tip):
                return True
    return False

@njit(fastmath=True)
def segment_segment_dist_sq(p1, p2, q1, q2):
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
        sN = 0.0; sD = 1.0; tN = e; tD = c
    else:
        sN = (b*e - c*d); tN = (a*e - b*d)
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
            if dist_sq < (r1 + r2 + 0.01)**2:
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

class SAGEPacker:
    def __init__(self, container_dims, h_unit=1.13):
        self.container_dims = np.array(container_dims)
        self.h_unit = h_unit
        self.r_base = 0.47 * h_unit / 2
        self.r_tip = 0.3 * h_unit / 2
        self.leg_h = 0.75 * h_unit
        
        self.orig_dirs = np.array([
            [0, 0, 1],
            [np.sqrt(8.0)/3.0, 0, -1.0/3.0],
            [-np.sqrt(2.0)/3.0, np.sqrt(6.0)/3.0, -1.0/3.0],
            [-np.sqrt(2.0)/3.0, -np.sqrt(6.0)/3.0, -1.0/3.0]
        ])
        
        # Dense point cloud for collision
        pts = []
        for d in self.orig_dirs:
            for z_f in np.linspace(0, 1, 10):
                z = z_f * self.leg_h
                r = self.r_base + z_f * (self.r_tip - self.r_base)
                for angle in np.linspace(0, 2*np.pi, 6):
                    p = np.array([r*np.cos(angle), r*np.sin(angle), z])
                    m = trimesh.geometry.align_vectors([0,0,1], d)
                    p_rot = trimesh.transformations.transform_points([p], m)[0]
                    pts.append(p_rot)
        self.point_cloud = np.array(pts)
        self.packed_items_data = []

    def pack_items(self, rotations):
        self.packed_items_data = []
        for rot in rotations:
            self.pack_item(rot)
        return len(self.packed_items_data)

    def pack_item(self, rotation):
        euler_matrix = trimesh.transformations.euler_matrix(*rotation)
        trans_dirs_4x4 = np.eye(4)
        trans_dirs_4x4[:3, :3] = euler_matrix[:3, :3]
        trans_dirs = transform_points_numba(self.orig_dirs, trans_dirs_4x4)
        trans_pts = transform_points_numba(self.point_cloud, euler_matrix)
        
        # Calculate bounds
        min_v = np.array([np.min(trans_pts[:,0]), np.min(trans_pts[:,1]), np.min(trans_pts[:,2])])
        max_v = np.array([np.max(trans_pts[:,0]), np.max(trans_pts[:,1]), np.max(trans_pts[:,2])])
        rotated_bounds = np.stack((min_v, max_v))

        # Heuristic search grid
        nx, ny = 12, 5
        x_steps = np.linspace(0.6, self.container_dims[0]-0.6, nx)
        y_steps = np.linspace(0.6, self.container_dims[1]-0.6, ny)
        
        best_pos = None
        min_z = float('inf')
        
        for x in x_steps:
            for y in y_steps:
                pos = self.drop_test(trans_pts, trans_dirs, rotated_bounds, x, y)
                if pos is not None:
                    # LOCAL SETTLEMENT (Wiggle)
                    pos = self.local_wiggle(trans_pts, trans_dirs, rotated_bounds, pos)
                    
                    if pos[2] < min_z:
                        min_z = pos[2]
                        best_pos = pos
                    if min_z < 0.7: break
            if min_z < 0.7: break
            
        if best_pos is not None:
            self.packed_items_data.append({
                'pts': trans_pts + best_pos,
                'bases': np.zeros((4, 3)) + best_pos,
                'dirs': trans_dirs,
                'bounds': rotated_bounds + best_pos,
                'rot': rotation,
                'pos': best_pos
            })
            return True
        return False

    def check_collision(self, pts, bases, dirs, bounds):
        for item in self.packed_items_data:
            pb = item['bounds']
            # Fast AABB
            if (pb[1,0] < bounds[0,0] or pb[0,0] > bounds[1,0] or
                pb[1,1] < bounds[0,1] or pb[0,1] > bounds[1,1] or
                pb[1,2] < bounds[0,2] or pb[0,2] > bounds[1,2]):
                continue
            
            # 1. Segment-Segment check
            if check_segment_collision(bases[0], dirs, item['bases'][0], item['dirs'], self.leg_h, self.r_base, self.r_tip):
                return True
                
            # 2. Point-in-Cone check
            if check_tetrapod_collision(pts, item['bases'], item['dirs'], self.leg_h, self.r_base, self.r_tip) or \
               check_tetrapod_collision(item['pts'], bases, dirs, self.leg_h, self.r_base, self.r_tip):
                return True
        return False

    def local_wiggle(self, pts, dirs, bounds, start_pos):
        curr_pos = start_pos.copy()
        for _ in range(20):
            nudge = (np.random.rand(3) - 0.5) * 0.1
            nudge[2] = -0.05
            test_pos = curr_pos + nudge
            test_pts = pts + test_pos
            test_bases = np.zeros((4,3)) + test_pos
            test_bounds = bounds + test_pos
            
            if np.any(test_pts[:,0] < 0) or np.any(test_pts[:,0] > self.container_dims[0]) or \
               np.any(test_pts[:,1] < 0) or np.any(test_pts[:,1] > self.container_dims[1]) or \
               np.any(test_pts[:,2] < 0):
                continue
                
            if not self.check_collision(test_pts, test_bases, dirs, test_bounds):
                curr_pos = test_pos
        return curr_pos

    def drop_test(self, pts, dirs, bounds, x, y):
        z_start = self.container_dims[2] - bounds[1,2]
        offset = np.array([x, y, z_start])
        curr_pts = pts + offset
        curr_bases = np.zeros((4, 3)) + offset
        
        if self.check_collision(curr_pts, curr_bases, dirs, bounds + offset):
            return None

        z_step = 0.4
        total_drop = 0
        while z_step > 0.02:
            drop = np.array([0, 0, -z_step])
            test_pts = curr_pts + drop
            test_bases = curr_bases + drop
            test_bounds = (bounds + offset) + np.array([0, 0, -total_drop - z_step])
            
            if test_bounds[0,2] < 0 or self.check_collision(test_pts, test_bases, dirs, test_bounds):
                z_step /= 2
            else:
                curr_pts = test_pts
                curr_bases = test_bases
                total_drop += z_step
                
        return offset + np.array([0, 0, -total_drop])

def optimize_packing(container_dims, iterations=30):
    packer = SAGEPacker(container_dims)
    best_count = 0
    best_rots = None
    
    print(f"Starting Advanced SAGE3D (Wiggle + Spatial) Optimization...")
    stable_rot = [np.arccos(-1.0/3.0), 0, 0]
    
    for i in range(iterations):
        test_rots = [stable_rot if np.random.rand() > 0.3 else (np.random.rand(3)*2*np.pi) for _ in range(100)]
        count = packer.pack_items(test_rots)
        if count >= best_count:
            best_count = count
            best_rots = test_rots
            print(f"Iteration {i+1}: Packed {best_count} units")
    return best_count, best_rots

if __name__ == "__main__":
    container_dims = (5.898, 2.352, 2.393)
    start_time = time.time()
    count, best_rots = optimize_packing(container_dims, iterations=20)
    end_time = time.time()
    print(f"\nAdvanced SAGE3D finished in {end_time - start_time:.2f}s. Final Count: {count}")
    packer = SAGEPacker(container_dims)
    packer.pack_items(best_rots)
    import generate_mesh
    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
    scene = trimesh.Scene()
    for item in packer.packed_items_data:
        m = tp_mesh.copy()
        m.apply_transform(trimesh.transformations.euler_matrix(*item['rot']))
        m.apply_translation(item['pos'])
        scene.add_geometry(m)
    container_box = trimesh.creation.box(extents=container_dims)
    container_box.apply_translation(np.array(container_dims)/2)
    container_box.visual.face_colors = [100, 100, 100, 40]
    scene.add_geometry(container_box)
    scene.export("advanced_packing_result.glb")
    print("Result saved to advanced_packing_result.glb")
