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
def cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@njit
def segment_segment_dist_sq_info(p1, p2, q1, q2):
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
def enforce_constraints(positions, quaternions, n, current_scale, iterations=30):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    rb = R_BASE * current_scale
    rt = R_TIP * current_scale
    
    for _it in range(iterations):
        # 1. Inter-Object Collision (Position + Rotation Push)
        for i in range(n):
            dirs_i = np.zeros((4, 3))
            for k in range(4): dirs_i[k] = q_rotate(quaternions[i], orig_dirs[k])
                
            for j in range(i + 1, n):
                dist_centers_sq = np.sum((positions[i] - positions[j])**2)
                if dist_centers_sq > (LEG_H * 3.0)**2: continue
                
                dirs_j = np.zeros((4, 3))
                for k in range(4): dirs_j[k] = q_rotate(quaternions[j], orig_dirs[k])
                
                for k1 in range(4):
                    p1, p2 = positions[i], positions[i] + dirs_i[k1] * LEG_H
                    for k2 in range(4):
                        q_start, q_end = positions[j], positions[j] + dirs_j[k2] * LEG_H
                        d_sq, t, u, diff = segment_segment_dist_sq_info(p1, p2, q_start, q_end)
                        r1 = rb + t * (rt - rb)
                        r2 = rb + u * (rt - rb)
                        target = r1 + r2 + 0.015 # Increased comfort gap (1.5cm)
                        if d_sq < target**2:
                            dist = np.sqrt(d_sq)
                            overlap = target - dist
                            push_dir = diff / (dist + 1e-9)
                            push_force = push_dir * (overlap * 0.5)
                            
                            # Position Update
                            positions[i] += push_force
                            positions[j] -= push_force
                            
                            # Rotation Update (Simple Torque Proxy)
                            lever_i = (p1 + t * (p2 - p1)) - positions[i]
                            torque_i = cross(lever_i, push_force)
                            dq_i = 0.5 * q_multiply(quaternions[i], np.array([0.0, torque_i[0], torque_i[1], torque_i[2]])) * 0.2
                            quaternions[i] += dq_i
                            quaternions[i] /= np.linalg.norm(quaternions[i])
                            
                            lever_j = (q_start + u * (q_end - q_start)) - positions[j]
                            torque_j = cross(lever_j, -push_force)
                            dq_j = 0.5 * q_multiply(quaternions[j], np.array([0.0, torque_j[0], torque_j[1], torque_j[2]])) * 0.2
                            quaternions[j] += dq_j
                            quaternions[j] /= np.linalg.norm(quaternions[j])

        # 2. IRON CONTAINER WALLS
        for i in range(n):
            dirs_i = np.zeros((4, 3))
            for k in range(4): dirs_i[k] = q_rotate(quaternions[i], orig_dirs[k])
            
            # Hub check
            for d in range(3):
                if positions[i, d] < rb + 0.01: positions[i, d] = rb + 0.01
                if positions[i, d] > REAL_CONTAINER[d] - rb - 0.01: positions[i, d] = REAL_CONTAINER[d] - rb - 0.01
            
            # Leg Samples (More dense)
            for k in range(4):
                for f in [0.25, 0.5, 0.75, 1.0]: 
                    point = positions[i] + dirs_i[k] * LEG_H * f
                    radius = rb + f * (rt - rb)
                    
                    correction = np.zeros(3)
                    for d in range(3):
                        if point[d] < radius + 0.01: 
                            correction[d] = max(correction[d], (radius + 0.01) - point[d])
                        if point[d] > REAL_CONTAINER[d] - radius - 0.01: 
                            correction[d] = min(correction[d], (REAL_CONTAINER[d] - radius - 0.01) - point[d])
                    
                    positions[i] += correction

def run_iron_stacker_v36(target_n=22, iterations=15000):
    print(f"Starting SAGE V36 (Angular Resolve, {target_n} units)...")
    # Initialize with slightly more room
    positions = np.random.rand(target_n, 3) * (REAL_CONTAINER - 1.4) + 0.7
    quaternions = np.zeros((target_n, 4))
    for i in range(target_n):
        q = np.random.rand(4) - 0.5
        quaternions[i] = q / np.linalg.norm(q)
        
    start_time = time.time()
    for step in range(iterations):
        progress = step / iterations
        
        # Physics Phase
        for i in range(target_n):
            # Gravity
            positions[i, 2] -= 0.008 * (1.0 - progress * 0.5)
            # Lateral squeeze to center
            positions[i, 0:2] += (REAL_CONTAINER[0:2]/2 - positions[i, 0:2]) * 0.001 * (1.0 - progress)
            
            # Agitation
            if progress < 0.85:
                # Random jitter
                positions[i] += (np.random.rand(3) - 0.5) * 0.02 * (1.0 - progress)
                dq = (np.random.rand(4) - 0.5) * 0.1 * (1.0 - progress)
                quaternions[i] = (quaternions[i] + dq) / np.linalg.norm(quaternions[i] + dq)
        
        # Constraint Solver
        # Increase internal iterations as we cool down
        solve_iters = 10 if progress < 0.5 else 20
        enforce_constraints(positions, quaternions, target_n, 1.0, iterations=solve_iters)
        
        if step % 2500 == 0:
            print(f"Step {step:5d} | System Pressure: {1.0 - progress:.2f}")

    return positions, quaternions, target_n

if __name__ == "__main__":
    try:
        # 22 units is the target for a stable, collision-free pack
        pos, qs, count = run_iron_stacker_v36(target_n=22, iterations=15000)
        
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            matrix = trimesh.transformations.quaternion_matrix(qs[i])
            matrix[:3, 3] = pos[i]
            scene.add_geometry(tp_mesh, transform=matrix, node_name=f"tp_{i}")
            
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 30]
        scene.add_geometry(container_box, node_name="container")
        
        output_path = "sage_tetrapod/export_results/36_IronContainer_Safe_Angular.glb"
        scene.export(output_path)
        print(f"V36 Result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
