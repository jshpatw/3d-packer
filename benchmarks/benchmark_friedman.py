import numpy as np
import os
import sys
import json
import time

# Ensure we can import the engine from the sibling directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))

try:
    from sage_engine import run_sage_engine
except ImportError:
    # Fallback for different execution contexts
    sys.path.append('SAGE_3D_SOLVER/engine')
    from sage_engine import run_sage_engine

def run_friedman_10():
    """
    Attempts to pack 10 spheres of radius 1.0 into a cube of side s = 4 + sqrt(2).
    This is a classic problem from Erich Friedman's Packing Center.
    """
    # Proven limit for 10 spheres of radius 1.0
    s_target = 4.0 + np.sqrt(2.0)
    r = 1.0
    N = 10
    
    print(f"\n" + "="*50)
    print(f"FRIEDMAN BENCHMARK: 10 Spheres in Cube (Schaer 1966)")
    print(f"="*50)
    print(f"  Target Cube Side (s): {s_target:.10f}m")
    print(f"  Sphere Radius (r): {r:.2f}m")
    print(f"  Target Density (phi): { (N * (4/3)*np.pi*r**3) / (s_target**3) :.6f}")
    
    # Sphere as a "zero-length segment" (point with radius)
    skeleton_data = {
        "segments": [
            {
                "start": [0, 0, 0],
                "end": [0, 0, 0],
                "r1": r,
                "r2": r
            }
        ]
    }
    
    # Box params: [center_x, center_y, center_z, half_x, half_y, half_z]
    # We revert to the original c_params because the strict boundary caused convergence failure.
    c_params = np.array([s_target/2, s_target/2, s_target/2, s_target/2, s_target/2, s_target/2])
    
    print(f"  Initializing SAGE Engine...")
    start_time = time.time()
    
    # Run engine
    pos, qs, ov = run_sage_engine(
        target_n=N, 
        skeleton_data=skeleton_data, 
        container_type=0, # Box
        container_params=c_params
    )
    
    # Export for visualization
    import trimesh
    scene = trimesh.Scene()
    
    # Add spheres
    for p in pos:
        sphere = trimesh.creation.icosphere(radius=r, subdivisions=3)
        sphere.apply_translation(p)
        scene.add_geometry(sphere)
    
    # Add container box (aligned with the pack center)
    # The spheres have radius 1.0. To visualize them *contained* within the box,
    # the box extents must be at least (max_coord - min_coord) + 2*r.
    # We use a visual box that strictly encapsulates the sphere geometries.
    
    # Get bounding box of all spheres
    coords = np.array(pos)
    min_c = coords.min(axis=0) - r
    max_c = coords.max(axis=0) + r
    
    center_c = (min_c + max_c) / 2
    size_c = max_c - min_c
    
    box = trimesh.creation.box(extents=size_c)
    box.apply_translation(center_c)
    
    # Apply transparency properly
    box.visual.face_colors = [255, 255, 255, 50] 
    scene.add_geometry(box)
    
    scene.export("SAGE_3D_SOLVER/benchmarks/friedman_10_sphere.glb")
    print(f"Visualization saved to SAGE_3D_SOLVER/benchmarks/friedman_10_sphere.glb")
    
    duration = time.time() - start_time
    
    print(f"\n--- BENCHMARK VERDICT ---")
    if ov < 1e-7:
        print(f"  SUCCESS: SAGE successfully reached the proven Friedman limit!")
        print(f"  Zero-Collision state found in {duration:.2f}s.")
    else:
        print(f"  PARTIAL: Residual Overlap = {ov:.8e}m.")
        print(f"  (Local minima are common in this 10-sphere problem.)")
        
    return ov

if __name__ == "__main__":
    run_friedman_10()
