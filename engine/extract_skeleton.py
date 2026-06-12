import os
import json
import numpy as np
import scipy.ndimage as ndimage
import trimesh

def extract_skeleton_from_stl(stl_path, pitch=0.04, min_radius_ratio=0.1):
    """
    Loads an STL mesh, centers it, voxelizes it, and extracts its curve skeleton
    as a list of segments: [ {start: [x,y,z], end: [x,y,z], r1: float, r2: float}, ... ]
    """
    print(f"Loading mesh: {stl_path}...")
    mesh = trimesh.load(stl_path)
    
    # 1. Center the mesh at its Center of Mass (COM)
    com = mesh.center_mass
    mesh.apply_translation(-com)
    
    # 2. Voxelize the mesh
    voxels = mesh.voxelized(pitch=pitch).fill()
    grid = voxels.matrix.astype(np.uint8)
    
    # 3. Compute Euclidean Distance Transform (EDT)
    edt = ndimage.distance_transform_edt(grid) * pitch
    max_thickness = edt.max()
    
    # 4. Extract Medial Axis points
    neighborhood_max = ndimage.maximum_filter(edt, size=3)
    threshold = max_thickness * min_radius_ratio
    ridge_mask = (grid > 0) & (edt == neighborhood_max) & (edt > threshold)
    
    z, y, x = np.where(ridge_mask)
    grid_min = voxels.bounds[0]
    points = np.column_stack((x, y, z)) * pitch + grid_min
    radii = edt[z, y, x]
    
    print(f"Found {len(points)} skeleton points. Max thickness: {max_thickness:.4f}")
    
    # 5. Simplify into segments
    # Find the center (closest to COM)
    dists_from_com = np.linalg.norm(points, axis=1)
    center_idx = np.argmin(dists_from_com)
    center_pt = points[center_idx]
    center_r = radii[center_idx]
    
    # Identify tip candidates: points far from center
    tip_threshold = max_thickness * 1.2
    tip_candidates_mask = dists_from_com > tip_threshold
    tip_points = points[tip_candidates_mask]
    tip_radii = radii[tip_candidates_mask]
    
    print(f"Tip threshold: {tip_threshold:.4f}. Found {len(tip_points)} tip candidate points.")
    
    segments = []
    if len(tip_points) > 0:
        # Cluster the tip points to find distinct branches
        visited = np.zeros(len(tip_points), dtype=bool)
        clusters = []
        eps = pitch * 4
        for i in range(len(tip_points)):
            if visited[i]: continue
            cluster = [i]
            visited[i] = True
            q = [i]
            while q:
                curr = q.pop(0)
                dists = np.linalg.norm(tip_points - tip_points[curr], axis=1)
                neighbors = np.where((dists < eps) & (~visited))[0]
                for n in neighbors:
                    visited[n] = True
                    cluster.append(n)
                    q.append(n)
            clusters.append(cluster)
        
        for cluster_indices in clusters:
            c_pts = tip_points[cluster_indices]
            c_radii = tip_radii[cluster_indices]
            # Furthest point in cluster from COM
            c_dists = np.linalg.norm(c_pts, axis=1)
            tip_idx = np.argmax(c_dists)
            
            segments.append({
                "start": center_pt.tolist(),
                "end": c_pts[tip_idx].tolist(),
                "r1": float(center_r),
                "r2": float(c_radii[tip_idx])
            })
            
    return {
        "source_mesh": os.path.basename(stl_path),
        "com_offset": com.tolist(),
        "segments": segments
    }

def main():
    # Use the 1-ton tetrapod STL as the test target
    stl_file = "sage_tetrapod/tetrapod_1ton.stl"
    
    if not os.path.exists(stl_file):
        # Fallback to the root copy if it's there
        stl_file = "tetrapod_1ton.stl"
        
    if os.path.exists(stl_file):
        output_json = "sage_tetrapod/extracted_skeleton.json"
        
        # Extract the skeleton
        skeleton = extract_skeleton_from_stl(stl_file, pitch=0.04)
        
        # Save to JSON
        with open(output_json, "w") as f:
            json.dump(skeleton, f, indent=4)
            
        print(f"\n--- SUCCESS ---")
        print(f"Skeleton successfully extracted and saved to: {output_json}")
        print(f"Extracted Segments (COM-aligned):")
        for i, seg in enumerate(skeleton["segments"]):
            print(f"  Segment {i}: {seg['start']} -> {seg['end']} (r={seg['r1']:.3f} to {seg['r2']:.3f})")
    else:
        print(f"Error: Could not find '{stl_file}' to process.")

if __name__ == "__main__":
    main()
