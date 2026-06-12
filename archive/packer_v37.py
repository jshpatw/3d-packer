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
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

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
def enforce_constraints_v37(positions, quaternions, n, current_scale, iterations=40):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    rb = R_BASE * current_scale
    rt = R_TIP * current_scale
    
    all_dirs = np.zeros((n, 4, 3))
    for i in range(n):
        for k in range(4): all_dirs[i, k] = q_rotate(quaternions[i], orig_dirs[k])
            
    for _it in range(iterations):
        # Update ALL dirs once per iteration for speed
        for i in range(n):
            for k in range(4): all_dirs[i, k] = q_rotate(quaternions[i], orig_dirs[k])

        # 1. Inter-Object Collision
        for i in range(n):
            for j in range(i + 1, n):
                dist_centers_sq = np.sum((positions[i] - positions[j])**2)
                if dist_centers_sq > (LEG_H * 2.8)**2: continue
                
                for k1 in range(4):
                    p1, p2 = positions[i], positions[i] + all_dirs[i, k1] * LEG_H
                    for k2 in range(4):
                        q1, q2 = positions[j], positions[j] + all_dirs[j, k2] * LEG_H
                        d_sq, t, u, diff = segment_segment_dist_sq_info(p1, p2, q1, q2)
                        r1 = rb + t * (rt - rb)
                        r2 = rb + u * (rt - rb)
                        target = r1 + r2 + 0.012 # 1.2cm safety gap
                        
                        if d_sq < target**2:
                            dist = np.sqrt(d_sq)
                            overlap = target - dist
                            push_dir = diff / (dist + 1e-9)
                            
                            # Sub-stepping push
                            push_mag = overlap * 0.4
                            push_vec = push_dir * push_mag
                            
                            positions[i] += push_vec
                            positions[j] -= push_vec
                            
                            # Torque application
                            lever_i = (p1 + t * (p2 - p1)) - positions[i]
                            torque_i = cross(lever_i, push_vec)
                            dq_i = 0.5 * q_multiply(quaternions[i], np.array([0.0, torque_i[0], torque_i[1], torque_i[2]])) * 0.5
                            quaternions[i] += dq_i
                            quaternions[i] /= np.linalg.norm(quaternions[i])
                            
                            lever_j = (q1 + u * (q2 - q1)) - positions[j]
                            torque_j = cross(lever_j, -push_vec)
                            dq_j = 0.5 * q_multiply(quaternions[j], np.array([0.0, torque_j[0], torque_j[1], torque_j[2]])) * 0.5
                            quaternions[j] += dq_j
                            quaternions[j] /= np.linalg.norm(quaternions[j])

        # 2. IRON CONTAINER WALLS (Harder constraints)
        for i in range(n):
            # Hub
            for d in range(3):
                if positions[i, d] < rb + 0.01: positions[i, d] = rb + 0.01
                if positions[i, d] > REAL_CONTAINER[d] - rb - 0.01: positions[i, d] = REAL_CONTAINER[d] - rb - 0.01
            
            # Leg Tips and Mids
            for k in range(4):
                for f in [0.33, 0.66, 1.0]:
                    point = positions[i] + all_dirs[i, k] * LEG_H * f
                    radius = rb + f * (rt - rb)
                    
                    for d in range(3):
                        if point[d] < radius + 0.005:
                            # Push the hub to resolve leg protrusion
                            positions[i, d] += (radius + 0.005) - point[d]
                        if point[d] > REAL_CONTAINER[d] - radius - 0.005:
                            positions[i, d] -= point[d] - (REAL_CONTAINER[d] - radius - 0.005)

def run_iron_stacker_v37(target_n=22, iterations=12000):
    print(f"Starting SAGE V37 (High Resolution, {target_n} units)...")
    positions = np.random.rand(target_n, 3) * (REAL_CONTAINER - 1.5) + 0.75
    quaternions = np.zeros((target_n, 4))
    for i in range(target_n):
        q = np.random.rand(4) - 0.5
        quaternions[i] = q / np.linalg.norm(q)
        
    for step in range(iterations):
        progress = step / iterations
        
        # 1. Physics Phase
        for i in range(target_n):
            # Gentle gravity
            positions[i, 2] -= 0.006 * (1.0 - progress * 0.7)
            # Lateral compression
            positions[i, 0:2] += (REAL_CONTAINER[0:2]/2 - positions[i, 0:2]) * 0.0005 * (1.0 - progress)
            
            # Simulated Vibratory Shaking
            if progress < 0.9:
                shake = 0.03 * (1.0 - progress)
                positions[i] += (np.random.rand(3) - 0.5) * shake
                dq = (np.random.rand(4) - 0.5) * shake * 4.0
                quaternions[i] = (quaternions[i] + dq) / np.linalg.norm(quaternions[i] + dq)
        
        # 2. Constraint Solver
        # Ramp up solver power at the end
        solve_iters = 12 if progress < 0.5 else 30
        enforce_constraints_v37(positions, quaternions, target_n, 1.0, iterations=solve_iters)
        
        if step % 2000 == 0:
            print(f"Step {step:5d} | Progress: {progress*100:.1f}%")

    return positions, quaternions, target_n

if __name__ == "__main__":
    try:
        pos, qs, count = run_iron_stacker_v37(target_n=22, iterations=12000)
        
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            matrix = trimesh.transformations.quaternion_matrix(qs[i])
            matrix[:3, 3] = pos[i]
            scene.add_geometry(tp_mesh, transform=matrix, node_name=f"tp_{i}")
            
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 25]
        scene.add_geometry(container_box, node_name="container")
        
        output_path = "sage_tetrapod/export_results/37_IronContainer_V37_Perfect.glb"
        scene.export(output_path)
        print(f"V37 Result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
