import trimesh
import numpy as np
import os
import time
from numba import njit

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
    return dist_sq <= (r_at_h + 0.01)**2

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
        self.orig_dirs = np.array([[0,0,1], [np.sqrt(8.0)/3.0,0,-1.0/3.0], [-np.sqrt(2.0)/3.0,np.sqrt(6.0)/3.0,-1.0/3.0], [-np.sqrt(2.0)/3.0,-np.sqrt(6.0)/3.0,-1.0/3.0]])
        pts = []
        for d in self.orig_dirs:
            for z_f in np.linspace(0, 1, 12):
                z, r = z_f * self.leg_h, self.r_base + z_f * (self.r_tip - self.r_base)
                for angle in np.linspace(0, 2*np.pi, 6):
                    p = np.array([r*np.cos(angle), r*np.sin(angle), z])
                    m = trimesh.geometry.align_vectors([0,0,1], d)
                    pts.append(trimesh.transformations.transform_points([p], m)[0])
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
        min_v = np.array([np.min(trans_pts[:,0]), np.min(trans_pts[:,1]), np.min(trans_pts[:,2])])
        max_v = np.array([np.max(trans_pts[:,0]), np.max(trans_pts[:,1]), np.max(trans_pts[:,2])])
        rotated_bounds = np.stack((min_v, max_v))
        nx, ny = 12, 5
        x_steps, y_steps = np.linspace(0.6, self.container_dims[0]-0.6, nx), np.linspace(0.6, self.container_dims[1]-0.6, ny)
        best_pos, min_z = None, float('inf')
        for x in x_steps:
            for y in y_steps:
                pos = self.drop_test(trans_pts, trans_dirs, rotated_bounds, x, y)
                if pos is not None:
                    if pos[2] < min_z:
                        min_z = pos[2]
                        best_pos = pos
                    if min_z < 0.8: break
            if min_z < 0.8: break
        if best_pos is not None:
            self.packed_items_data.append((np.zeros((4, 3)) + best_pos, trans_dirs, trans_pts + best_pos, rotated_bounds + best_pos, rotation, best_pos))
            return True
        return False

    def drop_test(self, pts, dirs, bounds, x, y):
        z_start = self.container_dims[2] - bounds[1,2]
        offset = np.array([x, y, z_start])
        curr_pts, curr_bases = pts + offset, np.zeros((4, 3)) + offset
        for pb, pd, pp, pbounds, _, _ in self.packed_items_data:
            if check_segment_collision(curr_bases[0], dirs, pb[0], pd, self.leg_h, self.r_base, self.r_tip) or \
               check_tetrapod_collision(curr_pts, pb, pd, self.leg_h, self.r_base, self.r_tip) or \
               check_tetrapod_collision(pp, curr_bases, dirs, self.leg_h, self.r_base, self.r_tip):
                return None
        z_step, total_drop = 0.4, 0
        while z_step > 0.02:
            drop = np.array([0, 0, -z_step])
            test_pts, test_bases = curr_pts + drop, curr_bases + drop
            if (bounds[0,2] + offset[2] - total_drop - z_step) < 0:
                z_step /= 2
                continue
            collision = False
            for pb, pd, pp, pbounds, _, _ in self.packed_items_data:
                if (pbounds[1,0] < (bounds[0,0]+offset[0]-0.05) or pbounds[0,0] > (bounds[1,0]+offset[0]+0.05) or
                    pbounds[1,1] < (bounds[0,1]+offset[1]-0.05) or pbounds[0,1] > (bounds[1,1]+offset[1]+0.05) or
                    pbounds[1,2] < (bounds[0,2]+offset[2]-total_drop-z_step-0.05) or pbounds[0,2] > (bounds[1,2]+offset[2]-total_drop-z_step+0.05)):
                    continue
                if check_segment_collision(test_bases[0], dirs, pb[0], pd, self.leg_h, self.r_base, self.r_tip) or \
                   check_tetrapod_collision(test_pts, pb, pd, self.leg_h, self.r_base, self.r_tip) or \
                   check_tetrapod_collision(pp, test_bases, dirs, self.leg_h, self.r_base, self.r_tip):
                    collision = True
                    break
            if collision: z_step /= 2
            else: curr_pts, curr_bases, total_drop = test_pts, test_bases, total_drop + z_step
        return offset + np.array([0, 0, -total_drop])

def optimize_packing(container_dims, iterations=20):
    packer = SAGEPacker(container_dims)
    best_count, best_rots = 0, None
    stable_rot = [np.arccos(-1.0/3.0), 0, 0]
    current_rots = [stable_rot if np.random.rand() > 0.4 else (np.random.rand(3)*2*np.pi) for _ in range(80)]
    for i in range(iterations):
        test_rots = [(np.random.rand(3)*2*np.pi if np.random.rand() < 0.1 else r) for r in current_rots]
        count = packer.pack_items(test_rots)
        if count >= best_count:
            best_count, best_rots, current_rots = count, test_rots, test_rots
            print(f"Iteration {i+1}: Packed {best_count} units")
        elif np.random.rand() < 0.05: current_rots = test_rots
    return best_count, best_rots

if __name__ == "__main__":
    container_dims = (5.898, 2.352, 2.393)
    start_time = time.time()
    count, best_rots = optimize_packing(container_dims, iterations=20)
    print(f"\nOptimization finished in {time.time() - start_time:.2f}s. Final Count: {count}")
    packer = SAGEPacker(container_dims)
    packer.pack_items(best_rots)
    import generate_mesh
    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
    scene = trimesh.Scene()
    for _, _, _, _, rot, pos in packer.packed_items_data:
        m = tp_mesh.copy()
        m.apply_transform(trimesh.transformations.euler_matrix(*rot))
        m.apply_translation(pos)
        scene.add_geometry(m)
    container_box = trimesh.creation.box(extents=container_dims)
    container_box.apply_translation(np.array(container_dims)/2)
    container_box.visual.face_colors = [100, 100, 100, 40]
    scene.add_geometry(container_box)
    scene.export("analytical_packing_result.glb")
    print("Result saved to analytical_packing_result.glb")
