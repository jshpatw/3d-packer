import os
import time
import numpy as np
import trimesh
from sage_engine import run_sage_engine
import generate_mesh
from diagnose_overlaps import check_overlap_robust

def save_result(pos, qs, n, folder, filename):
    scene = trimesh.Scene()
    for i in range(len(pos)):
        tp_mesh = generate_mesh.create_tetrapod(h=1.13)
        matrix = trimesh.transformations.quaternion_matrix(qs[i])
        matrix[:3, 3] = pos[i]
        scene.add_geometry(tp_mesh, transform=matrix, node_name=f"tp_{i}")
    
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, filename)
    scene.export(out_path)
    return out_path

def generate_batch(count=10):
    batch_dir = "sage_tetrapod/export_results/04_Frontier_Research/Batch_24"
    os.makedirs(batch_dir, exist_ok=True)
    
    results = []
    
    print(f"=== GENERATING BATCH OF {count} UNIQUE 24-UNIT PACKS ===")
    for i in range(count):
        print(f"\n--- Generating Solution {i+1}/{count} ---")
        start = time.time()
        # The engine uses np.random internally, so each call produces a unique result
        pos, qs, ov = run_sage_engine(target_n=24)
        duration = time.time() - start
        
        filename = f"Perfect_24_Sol_{i+1:02d}.glb"
        path = save_result(pos, qs, 24, batch_dir, filename)
        
        # Immediate Audit
        print(f"Auditing {filename}...")
        dims = (5.898, 2.352, 2.393)
        bv, col = check_overlap_robust(path, dims)
        
        results.append({
            'id': i+1,
            'overlap': ov,
            'collisions': col,
            'time': duration
        })

    print("\n=== BATCH SUMMARY ===")
    print(f"{'ID':<5} | {'Solver Ov':<12} | {'Audit Col':<10} | {'Time':<8}")
    print("-" * 45)
    for r in results:
        print(f"{r['id']:<5} | {r['overlap']:<12.8f} | {r['collisions']:<10} | {r['time']:<8.2f}s")

if __name__ == "__main__":
    generate_batch(10)
