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
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
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
def check_collision_single(pos, dirs, packed_pos, packed_dirs, num_packed, buffer=0.01):
    for i in range(num_packed):
        # AABB Centroid Prune
        if np.sum((pos - packed_pos[i])**2) > (LEG_H * 2.5)**2: continue
        
        # Check Leg Pairs
        for k1 in range(4):
            p1, p2 = pos, pos + dirs[k1] * LEG_H
            for k2 in range(4):
                q1, q2 = packed_pos[i], packed_pos[i] + packed_dirs[i, k2] * LEG_H
                dist_sq, sc, tc = segment_segment_dist_sq(p1, p2, q1, q2)
                r1 = R_BASE + sc * (R_TIP - R_BASE)
                r2 = R_BASE + tc * (R_TIP - R_BASE)
                if dist_sq < (r1 + r2 + buffer)**2:
                    return True
    return False

@njit
def is_outside_container(pos, dirs, buffer=0.005):
    # Check Hub
    for d in range(3):
        if pos[d] < R_BASE + buffer or pos[d] > REAL_CONTAINER[d] - R_BASE - buffer: return True
    # Check Tips
    for k in range(4):
        tip = pos + dirs[k] * LEG_H
        for d in range(3):
            if tip[d] < R_TIP + buffer or tip[d] > REAL_CONTAINER[d] - R_TIP - buffer: return True
    return False

@njit
def find_drop_position(packed_pos, packed_dirs, num_packed, max_tries=150):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    
    for _t in range(max_tries):
        # Random start at top
        x = R_BASE + np.random.rand() * (REAL_CONTAINER[0] - 2*R_BASE)
        y = R_BASE + np.random.rand() * (REAL_CONTAINER[1] - 2*R_BASE)
        z = REAL_CONTAINER[2] - R_BASE - 0.1
        
        # Random orientation
        q = np.random.rand(4) - 0.5
        q /= np.linalg.norm(q)
        
        dirs = np.zeros((4, 3))
        for k in range(4): dirs[k] = q_rotate(q, orig_dirs[k])
        
        if is_outside_container(np.array([x, y, z]), dirs) or \
           check_collision_single(np.array([x, y, z]), dirs, packed_pos, packed_dirs, num_packed):
            continue
            
        # Dropping down
        pos = np.array([x, y, z])
        z_step = 0.2
        while z_step > 0.002:
            test_pos = pos + np.array([0.0, 0.0, -z_step])
            if is_outside_container(test_pos, dirs) or \
               check_collision_single(test_pos, dirs, packed_pos, packed_dirs, num_packed):
                z_step /= 2.0
            else:
                pos = test_pos
        
        return pos, q, True
        
    return np.zeros(3), np.zeros(4), False

def run_tetris_packer(target_n=22):
    print(f"Starting SAGE V41 (Sequential Tetris Dropper, {target_n} units)...")
    packed_positions = np.zeros((target_n, 3))
    packed_quaternions = np.zeros((target_n, 4))
    packed_dirs = np.zeros((target_n, 4, 3))
    num_packed = 0
    
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])

    for i in range(target_n):
        pos, q, success = find_drop_position(packed_positions, packed_dirs, num_packed)
        if success:
            packed_positions[num_packed] = pos
            packed_quaternions[num_packed] = q
            for k in range(4): packed_dirs[num_packed, k] = q_rotate(q, orig_dirs[k])
            num_packed += 1
            print(f"  Packed unit {num_packed}/{target_n} at Z={pos[2]:.2f}")
        else:
            print(f"  FAILED to pack unit {i+1}. Ending early.")
            break
            
    return packed_positions[:num_packed], packed_quaternions[:num_packed], num_packed

if __name__ == "__main__":
    try:
        pos, qs, count = run_tetris_packer(target_n=22)
        
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            matrix = trimesh.transformations.quaternion_matrix(qs[i])
            matrix[:3, 3] = pos[i]
            scene.add_geometry(tp_mesh, transform=matrix, node_name=f"tp_{i}")
        
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 10]
        scene.add_geometry(container_box, node_name="container")
        
        output_path = "sage_tetrapod/export_results/41_IronContainer_V41_Tetris.glb"
        scene.export(output_path)
        print(f"V41 Result saved to {output_path} | Count: {count}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
