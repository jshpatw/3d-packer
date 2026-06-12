import trimesh
import numpy as np

def test_collision():
    tp = trimesh.load("tetrapod_1ton.stl")
    print(f"Is watertight: {tp.is_watertight}")
    
    tp2 = tp.copy()
    # Move it so it clearly overlaps
    tp2.apply_translation([0.1, 0, 0])
    
    # Test 1: trimesh.proximity.signed_distance
    dists = trimesh.proximity.signed_distance(tp, tp2.vertices)
    print(f"Min signed distance: {np.min(dists)}")
    print(f"Any inside (d < -1e-4): {np.any(dists < -1e-4)}")
    
    # Test 2: Bounding box overlap
    print(f"Bounds 1: \n{tp.bounds}")
    print(f"Bounds 2: \n{tp2.bounds}")

if __name__ == "__main__":
    test_collision()
