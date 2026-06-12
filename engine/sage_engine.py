import trimesh
import numpy as np
import os
import time
import json
from numba import njit, prange
try:
    from sdf_library import get_container_sdf, get_sdf_gradient
except ImportError:
    from sage_tetrapod.sdf_library import get_container_sdf, get_sdf_gradient

@njit
def q_multiply(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
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
    u = p2 - p1; v = q2 - q1; w = p1 - q1
    a, b, c, d, e = np.dot(u,u), np.dot(u,v), np.dot(v,v), np.dot(u,w), np.dot(v,w)
    D = a*c - b*b
    sN, sD = 0.0, D; tN, tD = 0.0, D
    if D < 1e-8: sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = (b*e - c*d), (a*e - b*d)
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

@njit(parallel=True)
def get_gradients_generalized(pos, qs, n, segments, c_type, c_params, gravity=0.0):
    num_segs = len(segments)
    pos_grads = np.zeros((n, 3))
    rot_grads = np.zeros((n, 3))
    overlap_array = np.zeros(n)
    
    # 1. Pre-calculate transformed segments
    t_p1 = np.zeros((n, num_segs, 3))
    t_p2 = np.zeros((n, num_segs, 3))
    for i in prange(n):
        for s in range(num_segs):
            t_p1[i, s] = q_rotate(qs[i], segments[s, 0:3]) + pos[i]
            t_p2[i, s] = q_rotate(qs[i], segments[s, 3:6]) + pos[i]

    # 2. Build Spatial Grid for O(N) lookup
    # Determine max particle extent for cell size
    # segments: [num_segs, 8] -> x1, y1, z1, x2, y2, z2, r1, r2
    max_extent = 0.0
    for s in range(num_segs):
        l_sq = (segments[s,0]-segments[s,3])**2 + (segments[s,1]-segments[s,4])**2 + (segments[s,2]-segments[s,5])**2
        l = np.sqrt(l_sq)
        r = max(segments[s,6], segments[s,7])
        extent = l + 2*r
        if extent > max_extent: max_extent = extent
    
    cell_size = max_extent + 0.1
    grid_res = int(np.ceil(5.0 / cell_size))
    grid_res = max(2, min(30, grid_res)) # Clamp to reasonable range
    cell_size = 5.0 / grid_res
    grid_origin = np.array([0.0, 0.0, 0.0])
    
    # Count particles per cell
    cell_counts = np.zeros((grid_res, grid_res, grid_res), dtype=np.int32)
    particle_cells = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        cx = int(max(0, min(grid_res-1, (pos[i, 0] - grid_origin[0]) / cell_size)))
        cy = int(max(0, min(grid_res-1, (pos[i, 1] - grid_origin[1]) / cell_size)))
        cz = int(max(0, min(grid_res-1, (pos[i, 2] - grid_origin[2]) / cell_size)))
        particle_cells[i] = [cx, cy, cz]
        cell_counts[cx, cy, cz] += 1
        
    # Build cell offsets (linearized grid)
    cell_offsets = np.zeros(grid_res**3 + 1, dtype=np.int32)
    current_offset = 0
    for x in range(grid_res):
        for y in range(grid_res):
            for z in range(grid_res):
                cell_offsets[x*grid_res*grid_res + y*grid_res + z] = current_offset
                current_offset += cell_counts[x, y, z]
    cell_offsets[-1] = current_offset
    
    # Fill particle indices into grid
    grid_particles = np.zeros(n, dtype=np.int32)
    grid_cursor = cell_offsets[:-1].copy()
    for i in range(n):
        cx, cy, cz = particle_cells[i]
        idx = cx*grid_res*grid_res + cy*grid_res + cz
        grid_particles[grid_cursor[idx]] = i
        grid_cursor[idx] += 1

    # 3. Parallel Optimization Loop
    for i in prange(n):
        # Gravity
        pos_grads[i, 2] -= gravity

        # Boundary Repulsion
        for s in range(num_segs):
            p1, p2 = t_p1[i, s], t_p2[i, s]
            r1, r2 = segments[s, 6], segments[s, 7]
            for pt, r in [(p1, r1), (p2, r2)]:
                dist_to_wall = get_container_sdf(pt, c_type, c_params)
                if dist_to_wall + r > 0:
                    depth = dist_to_wall + r
                    normal = get_sdf_gradient(pt, c_type, c_params)
                    force = -normal * depth * 20.0
                    pos_grads[i] += force
                    rot_grads[i] += np.cross(pt - pos[i], force)
                    overlap_array[i] += depth

        # Inter-unit repulsion (Grid-based neighbors)
        cx, cy, cz = particle_cells[i]
        for dx in range(-1, 2):
            nx = cx + dx
            if nx < 0 or nx >= grid_res: continue
            for dy in range(-1, 2):
                ny = cy + dy
                if ny < 0 or ny >= grid_res: continue
                for dz in range(-1, 2):
                    nz = cz + dz
                    if nz < 0 or nz >= grid_res: continue
                    
                    cell_idx = nx*grid_res*grid_res + ny*grid_res + nz
                    start_p = cell_offsets[cell_idx]
                    end_p = cell_offsets[cell_idx + 1]
                    
                    for p_ptr in range(start_p, end_p):
                        j = grid_particles[p_ptr]
                        if i == j: continue
                        
                        # Distance check
                        dx_v = pos[i, 0] - pos[j, 0]
                        dy_v = pos[i, 1] - pos[j, 1]
                        dz_v = pos[i, 2] - pos[j, 2]
                        d_sq_com = dx_v*dx_v + dy_v*dy_v + dz_v*dz_v
                        if d_sq_com > 4.0: continue
                        
                        for s1 in range(num_segs):
                            p1, p2 = t_p1[i, s1], t_p2[i, s1]
                            r1_start, r1_end = segments[s1, 6], segments[s1, 7]
                            for s2 in range(num_segs):
                                q1, q2 = t_p1[j, s2], t_p2[j, s2]
                                r2_start, r2_end = segments[s2, 6], segments[s2, 7]
                                d_sq, sc, tc = segment_segment_dist_sq(p1, p2, q1, q2)
                                target_r = (r1_start + sc*(r1_end-r1_start)) + (r2_start + tc*(r2_end-r2_start))
                                if d_sq < target_r**2:
                                    dist = np.sqrt(d_sq)
                                    depth = target_r - dist
                                    overlap_array[i] += depth * 0.5
                                    normal = np.array([0.0, 0.0, 1.0]) if dist < 1e-8 else ( (p1 + sc*(p2-p1)) - (q1 + tc*(q2-q1)) ) / dist
                                    force = normal * depth * 10.0
                                    pos_grads[i] += force
                                    rot_grads[i] += np.cross(sc*(p2-p1), force)

    return np.sum(overlap_array), pos_grads, rot_grads

@njit
def apply_grads(pos, qs, pos_grads, rot_grads, lr):
    for i in range(len(pos)):
        f_norm = np.linalg.norm(pos_grads[i])
        if f_norm > 10.0: pos_grads[i] *= 10.0 / f_norm
        pos[i] += pos_grads[i] * lr
        w = rot_grads[i] * lr
        w_norm = np.linalg.norm(w)
        if w_norm > 0.5: w *= 0.5 / w_norm
        dq = np.array([1.0, w[0], w[1], w[2]])
        qs[i] = q_multiply(dq, qs[i])
        qs[i] /= np.linalg.norm(qs[i])

def run_gradient_engine(pos, qs, n, segments, c_type, c_params, steps=1000, lr=0.01, jitter=0.001, gravity=0.0):
    best_ov = 999.0
    for s in range(steps):
        ov, pg, rg = get_gradients_generalized(pos, qs, n, segments, c_type, c_params, gravity=gravity)
        if ov < best_ov: best_ov = ov
        apply_grads(pos, qs, pg, rg, lr)
        if jitter > 0:
            for i in range(n):
                pos[i] += (np.random.rand(3) - 0.5) * jitter
                dq = (np.random.rand(4) - 0.5) * jitter * 2.0
                qs[i] = (qs[i] + dq) / np.linalg.norm(qs[i] + dq)
        
        if s % 500 == 0:
            print("  Step", s, "| Overlap:", ov)
            
        if ov < 1e-6 and gravity == 0: return ov, s
    return best_ov, steps

def run_sage_engine(target_n=23, skeleton_data=None, container_type=0, container_params=None):
    if skeleton_data is None: return None
    segments = np.array([seg["start"] + seg["end"] + [seg["r1"], seg["r2"]] for seg in skeleton_data["segments"]])
    if container_params is None: container_params = np.array([2.949, 1.176, 1.1965, 2.949, 1.176, 1.1965])
    print(f"--- SAGE ENGINE v2.3 (Spatial Grid O(N) Mode) ---")
    print(f"Units: {target_n} | Container: {['Box', 'Cylinder', 'Sphere'][container_type]}")
    positions = np.zeros((target_n, 3))
    quaternions = np.zeros((target_n, 4))
    if container_type == 0:
        center = container_params[0:3]; half = container_params[3:6]
        low = center - half + 0.1; high = center + half - 0.1
    else:
        low, high = np.array([0.5, 0.5, 0.5]), np.array([5.0, 2.0, 2.0])
    side = int(np.ceil(target_n**(1/3))); idx = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if idx >= target_n: break
                u, v, w = i/max(1,side-1), j/max(1,side-1), k/max(1,side-1)
                positions[idx] = low + np.array([u,v,w]) * (high-low)
                quaternions[idx] = [1,0,0,0]
                quaternions[idx] += (np.random.rand(4)-0.5)*0.1; quaternions[idx] /= np.linalg.norm(quaternions[idx])
                idx += 1
    start_time = time.time()
    print("Phase 1: Global Stochastic Search...")
    run_gradient_engine(positions, quaternions, target_n, segments, container_type, container_params, steps=2000, lr=0.05, jitter=0.01)
    print("Phase 2: Analytical Gradient Squeeze...")
    run_gradient_engine(positions, quaternions, target_n, segments, container_type, container_params, steps=8000, lr=0.01, jitter=0.0001)
    print("Phase 3: Stability Audit (Gravity)...")
    pre_grav = positions.copy()
    ov, _ = run_gradient_engine(positions, quaternions, target_n, segments, container_type, container_params, steps=2000, lr=0.005, jitter=0.0, gravity=0.2)
    rmsd = np.sqrt(np.mean(np.sum((positions - pre_grav)**2, axis=1)))
    duration = time.time() - start_time
    print(f"--- Finished in {duration:.2f}s | Final Overlap: {ov:.8f}m ---")
    print(f"--- Stability Audit: RMSD = {rmsd:.6f}m ---")
    return positions, quaternions, ov

if __name__ == "__main__":
    skeleton_path = "sage_tetrapod/extracted_skeleton.json"
    with open(skeleton_path, "r") as f: skel = json.load(f)
    c_type = 1; c_params = np.array([2.949, 1.176, 1.1965, 1.1, 5.8])
    pos, qs, ov = run_sage_engine(target_n=20, skeleton_data=skel, container_type=c_type, container_params=c_params)
    scene = trimesh.Scene()
    for i in range(len(pos)):
        unit_group = []
        for seg in skel["segments"]:
            p1, p2 = np.array(seg["start"]), np.array(seg["end"])
            vec = p2 - p1; length = np.linalg.norm(vec)
            if length < 1e-6: continue
            cyl = trimesh.creation.cylinder(radius=seg["r1"], height=length)
            rotation = trimesh.geometry.align_vectors([0, 0, 1], vec); cyl.apply_transform(rotation)
            cyl.apply_translation(p1 + vec/2); unit_group.append(cyl)
        unit_mesh = trimesh.util.concatenate(unit_group)
        matrix = trimesh.transformations.quaternion_matrix(qs[i]); matrix[:3, 3] = pos[i]
        scene.add_geometry(unit_mesh, transform=matrix, node_name=f"unit_{i}")
    os.makedirs("sage_tetrapod/export_results/05_Generalized_Tests", exist_ok=True)
    out = "sage_tetrapod/export_results/05_Generalized_Tests/Cylinder_Pack_20.glb"
    scene.export(out); print(f"Result saved to {out}")
