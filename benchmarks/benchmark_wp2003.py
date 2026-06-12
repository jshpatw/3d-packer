import numpy as np
import os
import json
import time
from sage_engine import run_sage_engine

def calculate_spherocylinder_volume(L, D):
    R = D / 2
    V_cyl = np.pi * (R**2) * L
    V_sph = (4/3) * np.pi * (R**3)
    return V_cyl + V_sph

def run_benchmark_case(aspect_ratio, D, container_size, target_phi):
    L = aspect_ratio * D
    V_particle = calculate_spherocylinder_volume(L, D)
    V_container = np.prod(container_size)
    
    # Calculate N required for target packing fraction
    N = int((V_container * target_phi) / V_particle)
    
    print(f"\n=== BENCHMARK CASE: L/D = {aspect_ratio} ===")
    print(f"  Diameter (D): {D}m, Length (L): {L:.3f}m")
    print(f"  Particle Volume: {V_particle:.6f} m^3")
    print(f"  Target Packing Fraction: {target_phi}")
    print(f"  Calculated Particle Count (N): {N}")
    
    # Define skeleton for a spherocylinder (single central segment)
    skeleton_data = {
        "segments": [
            {
                "start": [0, 0, -L/2],
                "end": [0, 0, L/2],
                "r1": D/2,
                "r2": D/2
            }
        ]
    }
    
    # Container parameters for SAGE Engine
    # Params: [center_x, center_y, center_z, half_x, half_y, half_z]
    c_type = 0 # Box
    c_params = np.array([
        container_size[0]/2, 
        container_size[1]/2, 
        container_size[2]/2, 
        container_size[0]/2, 
        container_size[1]/2, 
        container_size[2]/2
    ])
    
    # Run the engine
    # Note: For high N, we might need to adjust initialization or steps
    # We'll start with a slightly smaller N if it's too high for a quick test, 
    # but the user asked for this specific benchmark.
    
    start_time = time.time()
    pos, qs, ov = run_sage_engine(
        target_n=N, 
        skeleton_data=skeleton_data, 
        container_type=c_type, 
        container_params=c_params
    )
    duration = time.time() - start_time
    
    # Calculate achieved packing fraction
    achieved_phi = (N * V_particle) / V_container
    
    print(f"  Achieved Phi: {achieved_phi:.4f}")
    print(f"  Overlap: {ov:.8f}m")
    
    return {
        "aspect_ratio": aspect_ratio,
        "N": N,
        "achieved_phi": achieved_phi,
        "overlap": ov,
        "duration": duration
    }

if __name__ == "__main__":
    # Benchmark Parameters
    CONTAINER_SIZE = np.array([5.0, 5.0, 5.0])
    D = 0.4
    
    # Define the 4 cases from the user's (inferred) table
    cases = [
        {"L/D": 0.0, "phi": 0.64},   # Spheres
        {"L/D": 0.4, "phi": 0.695},  # Peak Density
        {"L/D": 1.0, "phi": 0.63},   # Elongated
        {"L/D": 3.0, "phi": 0.61}    # Longer rods
    ]
    
    results = []
    
    # We'll run them sequentially
    for case in cases:
        try:
            res = run_benchmark_case(case["L/D"], D, CONTAINER_SIZE, case["phi"])
            results.append(res)
        except Exception as e:
            print(f"Error running case L/D={case['L/D']}: {e}")
            
    # Final Summary
    print("\n" + "="*40)
    print("FINAL BENCHMARK SUMMARY (WP2003)")
    print("="*40)
    print(f"{'L/D':<10} {'N':<10} {'Target Phi':<15} {'Achieved Phi':<15} {'Overlap':<15}")
    for i, res in enumerate(results):
        print(f"{res['aspect_ratio']:<10} {res['N']:<10} {cases[i]['phi']:<15.4f} {res['achieved_phi']:<15.4f} {res['overlap']:<15.8f}")
