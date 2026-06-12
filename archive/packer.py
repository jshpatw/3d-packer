import trimesh
import numpy as np
import os
import time

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

class SAGEPacker:
    def __init__(self, container_dims, h_unit=1.13):
        self.container_dims = np.array(container_dims)
        self.h_unit = h_unit
        self.r_base = 0.47 * h_unit / 2
        self.r_tip = 0.3 * h_unit / 2
        self.leg_h = 0.75 * h_unit
        self.orig_dirs = np.array([[0,0,1], [np.sqrt(8.0)/3.0,0,-1.0/3.0], [-np.sqrt(2.0)/3.0,np.sqrt(6.0)/3.0,-1.0/3.0], [-np.sqrt(2.0)/3.0,-np.sqrt(6.0)/3.0,-1.0/3.0]])
        self.packed_items = []
        
    def check_collision(self, pos, dirs):
        for p_pos, p_dirs, p_bounds in self.packed_items:
            # Segment-segment check
            collision = False
            for i in range(4):
                l1_p1 = pos
                l1_p2 = pos + dirs[i] * self.leg_h
                for j in range(4):
                    l2_p1 = p_pos
                    l2_p2 = p_pos + p_dirs[j] * self.leg_h
                    dist_sq, t, u = segment_segment_dist_sq(l1_p1, l1_p2, l2_p1, l2_p2)
                    r1 = self.r_base + t * (self.r_tip - self.r_base)
                    r2 = self.r_base + u * (self.r_tip - self.r_base)
                    if dist_sq < (r1 + r2 + 0.015)**2:
                        collision = True
                        break
                if collision: break
            if collision: return True
        return False

    def pack_items(self, rotations):
        self.packed_items = []
        for rot in rotations:
            self.pack_item(rot)
        return len(self.packed_items)

    def pack_item(self, rotation):
        mat = trimesh.transformations.euler_matrix(*rotation)
        dirs = self.orig_dirs @ mat[:3, :3].T
        
        # Approximate bounds for search
        x_steps = np.linspace(0.6, self.container_dims[0]-0.6, 10)
        y_steps = np.linspace(0.6, self.container_dims[1]-0.6, 5)
        
        best_pos = None
        min_z = float('inf')
        
        for x in x_steps:
            for y in y_steps:
                pos = self.drop_test(dirs, x, y)
                if pos is not None:
                    if pos[2] < min_z:
                        min_z = pos[2]
                        best_pos = pos
                    if min_z < 0.9: break
            if min_z < 0.9: break
        
        if best_pos is not None:
            self.packed_items.append((best_pos, dirs, None)) # Bounds not strictly needed here
            return True
        return False

    def drop_test(self, dirs, x, y):
        z_curr = self.container_dims[2] - 0.5
        pos = np.array([x, y, z_curr])
        
        if self.check_collision(pos, dirs):
            return None

        z_step = 0.5
        while z_step > 0.05:
            test_pos = pos + np.array([0, 0, -z_step])
            if test_pos[2] < 0.4 or self.check_collision(test_pos, dirs):
                z_step /= 2
            else:
                pos = test_pos
        return pos

def optimize_packing(container_dims, iterations=30):
    packer = SAGEPacker(container_dims)
    best_count = 0
    best_rots = None
    stable_rot = [np.arccos(-1/3), 0, 0]
    
    current_rots = [stable_rot if np.random.rand() > 0.4 else (np.random.rand(3)*2*np.pi) for _ in range(50)]
    
    for i in range(iterations):
        test_rots = [(np.random.rand(3)*2*np.pi if np.random.rand() < 0.1 else r) for r in current_rots]
        count = packer.pack_items(test_rots)
        if count >= best_count:
            best_count, best_rots, current_rots = count, test_rots, test_rots
            print(f"Iteration {i+1}: Packed {best_count}")
    return best_count, best_rots

if __name__ == "__main__":
    container_dims = (5.898, 2.352, 2.393)
    count, rots = optimize_packing(container_dims, iterations=20)
    print(f"Final Count: {count}")
    
    packer = SAGEPacker(container_dims)
    packer.pack_items(rots)
    
    import generate_mesh
    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
    scene = trimesh.Scene()
    for pos, dirs, _ in packer.packed_items:
        # Find rotation from original dirs to these dirs
        # For simplicity, we can just use the rots we saved if we had them.
        # But we'll just pack again and save properly.
        pass

    # Actually, the user should use packer_massive or packer_advanced.
    # I'll leave packer.py as a simplified reference.
