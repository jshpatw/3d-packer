import os
import time
import numpy as np
import trimesh
from sage_engine import run_sage_engine, REAL_CONTAINER
import generate_mesh
from diagnose_overlaps import check_overlap_robust

def save_result(pos, qs, n, filename):
    scene = trimesh.Scene()
    for i in range(len(pos)):
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        matrix = trimesh.transformations.quaternion_matrix(qs[i])
        matrix[:3, 3] = pos[i]
        scene.add_geometry(tp_mesh, transform=matrix, node_name=f"tp_{i}")
    
    out_dir = "sage_tetrapod/export_results/04_Frontier_Research"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    scene.export(out_path)
    return out_path

def run_stress_test():
    # 1. Generate more 24-unit variations to prove consistency
    print("\n=== GENERATING ADDITIONAL 24-UNIT PEAKS ===")
    for i in range(2):
        print(f"\n--- Variation {i+1} ---")
        pos, qs, ov = run_sage_engine(target_n=24)
        save_result(pos, qs, 24, f"Peak_24_Var_{i+1}.glb")

    # 2. Attempt 25 units (The "Impossible" push)
    print("\n=== ATTEMPTING 25 UNITS (STRESS TEST) ===")
    pos25, qs25, ov25 = run_sage_engine(target_n=25)
    path25 = save_result(pos25, qs25, 25, "Stress_Test_25_Attempt.glb")
    
    # 3. Attempt 26 units
    print("\n=== ATTEMPTING 26 UNITS (STRESS TEST) ===")
    pos26, qs26, ov26 = run_sage_engine(target_n=26)
    path26 = save_result(pos26, qs26, 26, "Stress_Test_26_Attempt.glb")

    print("\n=== FINAL LIMIT AUDIT ===")
    dims = (5.898, 2.352, 2.393)
    for p in ["Stress_Test_25_Attempt.glb", "Stress_Test_26_Attempt.glb"]:
        full_p = os.path.join("sage_tetrapod/export_results/04_Frontier_Research", p)
        print(f"\nAuditing {p}:")
        check_overlap_robust(full_p, dims)

if __name__ == "__main__":
    run_stress_test()
