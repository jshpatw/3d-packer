import numpy as np
import trimesh
from numba import njit
import trimesh.collision

# --- GROUND TRUTH KERNELS (Numba Optimized) ---
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
    return dist_sq <= (r_at_h + 0.002)**2

@njit(fastmath=True)
def check_tetrapod_collision(p_points, q_bases, q_dirs, q_h, q_r_base, q_r_tip):
    for i in range(p_points.ssage[0]):
        p = p_points[i]
        for j in range(4):
            if is_point_in_cone(p, q_bases[j], q_dirs[j], q_h, q_r_base, q_r_tip):
                return True
    return False

# --- DATA GENERATION ---
def get_tetrapod_data(h_unit=1.13):
    r_base = 0.47 * h_unit / 2
    r_tip = 0.3 * h_unit / 2
    leg_h = 0.75 * h_unit
    orig_dirs = np.array([
        [0,0,1], 
        [np.sqrt(8.0)/3.0, 0, -1.0/3.0], 
        [-np.sqrt(2.0)/3.0, np.sqrt(6.0)/3.0, -1.0/3.0], 
        [-np.sqrt(2.0)/3.0, -np.sqrt(6.0)/3.0, -1.0/3.0]
    ])

    pts = []
    for d in orig_dirs:
        for z_f in np.linspace(0, 1, 20):
            z, r = z_f * leg_h, r_base + z_f * (r_tip - r_base)
            for angle in np.linspace(0, 2*np.pi, 8):
                p = np.array([r*np.cos(angle), r*np.sin(angle), z])
                m = trimesh.geometry.align_vectors([0,0,1], d)
                pts.append(trimesh.transformations.transform_points([p], m)[0])
    
    pts = np.array(pts)
    return pts, orig_dirs, leg_h, r_base, r_tip

# --- COLLISION TEST SIMULATION ---
def test_repro_mesh_collision():
    print("Loading mesh and preparing data...")
    try:
        tetrapod_mesh = trimesh.load('tetrapod_1ton.stl')
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    pts, dirs, leg_h, r_base, r_tip = get_tetrapod_data()
    print(f"Base data ready. Starting 1000 trials...")

    miss_count = 0
    total_trials = 1000

    for i in range(total_trials):
        # Random pose for tetrapod 2
        pos2 = (np.random.rand(3) - 0.5) * 11.0 
        rot2 = np.random.rand(3) * 2 * np.pi
        mat2 = trimesh.transformations.euler_matrix(*rot2)

        # Transforms
        transform1 = np.eye(4) 
        transform2 = np.eye(4)
        transform2[:3, :3] = mat2[:3, :3] 
        transform2[:3, 3] = pos2 

        # Transformed Meshes
        m1 = tetrapod_mesh.copy().apply_transform(transform1)
        m2 = tetrapod_mesh.copy().apply_transform(transform2)

        # 1. Trimesh Collision Manager (Primary Check)
        manager = trimesh.collision.CollisionManager()
        manager.add_object('t1', m1)
        manager.add_object('t2', m2)
        hit_mesh = manager.in_contact(mesh_names=['t1', 't2'])

        # 2. Point-in-Cone Ground Truth
        trans_pts1 = trimesh.transformations.transform_points(pts, transform1)
        trans_dirs1 = dirs @ transform1[:3, :3].T # Correction: Use transpose for rotation
        trans_bases1 = np.zeros((4, 3)) + transform1[:3, 3]

        trans_pts2 = trimesh.transformations.transform_points(pts, transform2)
        trans_dirs2 = dirs @ transform2[:3, :3].T # Correction: Use transpose for rotation
        trans_bases2 = np.zeros((4, 3)) + transform2[:3, 3]

        ground_truth = check_tetrapod_collision(trans_pts1, trans_bases2, trans_dirs2, leg_h, r_base, r_tip) or \
                       check_tetrapod_collision(trans_pts2, trans_bases1, trans_dirs1, leg_h, r_base, r_tip)

        if not hit_mesh and ground_truth:
            miss_count += 1

        if (i+1) % 100 == 0:
            print(f"Trial {i+1} completed...")

    print(f"\nSimulation Finished.")
    print(f"Total trials: {total_trials}")
    print(f"Mesh intersection missed {miss_count} collisions that point check caught.")

if __name__ == "__main__":
    test_repro_mesh_collision()
