import trimesh
import numpy as np
import os
import generate_mesh

# Constants from sage_engine.py
REAL_CONTAINER = np.array([5.898, 2.352, 2.393])

def export_initial_state(target_n=23):
    print(f"Generating initial state for {target_n} units...")
    positions = np.zeros((target_n, 3))
    quaternions = np.zeros((target_n, 4))
    
    # Grid Initialization (Logic from sage_engine.py)
    idx = 0
    for layer in range(2):
        for row in range(2):
            for col in range(6):
                if idx >= target_n: break
                # These offsets and spacings are the "Egg Carton" logic
                positions[idx] = [0.6 + col * 0.93, 0.6 + row * 1.15, 0.5 + layer * 1.1]
                quaternions[idx] = [1, 0, 0, 0] # "Face North" Identity Quaternion
                idx += 1

    # Create Scene
    scene = trimesh.Scene()
    
    # Add Tetrapods
    tp_mesh = generate_mesh.create_tetrapod(h=1.13)
    for i in range(len(positions)):
        matrix = trimesh.transformations.quaternion_matrix(quaternions[i])
        matrix[:3, 3] = positions[i]
        # Use a distinct color for the initial state (e.g., semi-transparent red to show collisions)
        color = [200, 50, 50, 150] # Reddish
        mesh_copy = tp_mesh.copy()
        mesh_copy.visual.face_colors = color
        scene.add_geometry(mesh_copy, transform=matrix, node_name=f"tp_init_{i}")

    # Add Container Wireframe for context
    container_box = trimesh.creation.box(extents=REAL_CONTAINER)
    container_box.apply_translation(REAL_CONTAINER/2)
    # Convert to wireframe/path for better visibility of the units inside
    wireframe = container_box.outline()
    scene.add_geometry(wireframe, node_name="container_bounds")

    # Export
    os.makedirs("sage_tetrapod/export_results/paper_figures", exist_ok=True)
    out_path = "sage_tetrapod/export_results/paper_figures/Fig2_Initial_State_Grid.glb"
    scene.export(out_path)
    print(f"Initial state visualization saved to: {out_path}")

if __name__ == "__main__":
    export_initial_state(23)
