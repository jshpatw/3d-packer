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
def enforce_constraints(positions, quaternions, n, current_scale, iterations=25):
    orig_dirs = np.array([[0.0, 0.0, 1.0], [0.9428, 0.0, -0.3333], [-0.4714, 0.8165, -0.3333], [-0.4714, -0.8165, -0.3333]])
    rb = R_BASE * current_scale
    rt = R_TIP * current_scale
    
    for _it in range(iterations):
        # 1. Inter-Object Collision (Position Push)
        for i in range(n):
            dirs_i = np.zeros((4, 3))
            for k in range(4): dirs_i[k] = q_rotate(quaternions[i], orig_dirs[k])
                
            for j in range(i + 1, n):
                dist_centers_sq = np.sum((positions[i] - positions[j])**2)
                if dist_centers_sq > (LEG_H * 2.8)**2: continue
                
                dirs_j = np.zeros((4, 3))
                for k in range(4): dirs_j[k] = q_rotate(quaternions[j], orig_dirs[k])
                
                for k1 in range(4):
                    p1, p2 = positions[i], positions[i] + dirs_i[k1] * LEG_H
                    for k2 in range(4):
                        q_start, q_end = positions[j], positions[j] + dirs_j[k2] * LEG_H
                        d_sq, t, u, diff = segment_segment_dist_sq_info(p1, p2, q_start, q_end)
                        r1 = rb + t * (rt - rb)
                        r2 = rb + u * (rt - rb)
                        target = r1 + r2 + 0.005 # 5mm comfort gap
                        if d_sq < target**2:
                            dist = np.sqrt(d_sq)
                            overlap = target - dist
                            push = (diff / (dist + 1e-9)) * (overlap * 0.5)
                            positions[i] += push
                            positions[j] -= push

        # 2. IRON CONTAINER WALLS (ABSOLUTE FINAL SNAP)
        for i in range(n):
            dirs_i = np.zeros((4, 3))
            for k in range(4): dirs_i[k] = q_rotate(quaternions[i], orig_dirs[k])
            
            # Dense backbone check (13 points)
            for _sub in range(3):
                # Check Hub
                for d in range(3):
                    if positions[i, d] < rb + 0.02: positions[i, d] = rb + 0.02
                    if positions[i, d] > REAL_CONTAINER[d] - rb - 0.02: positions[i, d] = REAL_CONTAINER[d] - rb - 0.02
                
                # Check Leg Samples
                for k in range(4):
                    for f in [0.33, 0.66, 1.0]: # Mid points and Tip
                        point = positions[i] + dirs_i[k] * LEG_H * f
                        radius = rb + f * (rt - rb)
                        
                        correction = np.zeros(3)
                        for d in range(3):
                            if point[d] < radius + 0.02: 
                                correction[d] = max(correction[d], (radius + 0.02) - point[d])
                            if point[d] > REAL_CONTAINER[d] - radius - 0.02: 
                                correction[d] = min(correction[d], (REAL_CONTAINER[d] - radius - 0.02) - point[d])
                        
                        # Apply to hub to pull the sample point back inside
                        positions[i] += correction
    return 0

def run_iron_stacker(target_n=22, iterations=10000):
    print(f"Starting SAGE V35 (Iron Container, {target_n} units)...")
    positions = np.random.rand(target_n, 3) * (REAL_CONTAINER - 1.2) + 0.6
    quaternions = np.zeros((target_n, 4))
    for i in range(target_n):
        q = np.random.rand(4) - 0.5
        quaternions[i] = q / np.linalg.norm(q)
        
    start_time = time.time()
    for step in range(iterations):
        progress = step / iterations
        
        # 1. Physics
        for i in range(target_n):
            # Stronger gravity to force density
            positions[i, 2] -= 0.005 * (1.0 - progress)
            # Agitation
            if progress < 0.8:
                positions[i] += (np.random.rand(3) - 0.5) * 0.015 * (1.0 - progress)
                dq = (np.random.rand(4) - 0.5) * 0.08 * (1.0 - progress)
                quaternions[i] = (quaternions[i] + dq) / np.linalg.norm(quaternions[i] + dq)
        
        # 2. Iron Constraint Solver
        enforce_constraints(positions, quaternions, target_n, 1.0, iterations=10)
        
        if step % 2000 == 0:
            print(f"Step {step:5d} | Cooling Physics...")

    return positions, quaternions, target_n

if __name__ == "__main__":
    try:
        # Targeting 22 units for a dense but safe solve
        pos, qs, count = run_iron_stacker(target_n=22, iterations=12000)
        
        import generate_mesh
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        scene = trimesh.Scene()
        for i in range(count):
            m = tp_mesh.copy()
            matrix = trimesh.transformations.quaternion_matrix(qs[i])
            matrix[:3, 3] = pos[i]
            m.apply_transform(matrix)
            scene.add_geometry(m)
            
        container_box = trimesh.creation.box(extents=REAL_CONTAINER)
        container_box.apply_translation(REAL_CONTAINER/2)
        container_box.visual.face_colors = [100, 100, 100, 40]
        scene.add_geometry(container_box)
        
        output_path = "sage_tetrapod/export_results/35_IronContainer_Safe.glb"
        scene.export(output_path)
        print(f"Iron Container Result saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}"); import traceback; traceback.print_exc()
