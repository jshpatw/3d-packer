import sys
sys.path.append('SAGE_3D_SOLVER/engine')
from sage_engine import run_sage_engine
import trimesh
import numpy as np
import json

# Load tetrapod skeleton for segment data
with open('SAGE_3D_SOLVER/assets/extracted_skeleton.json', 'r') as f:
    skel = json.load(f)

# Run engine for 23 units in a rectangular container
# Correct dimensions for 20ft-like crate
container_dims = np.array([5.898, 2.352, 2.393])
# Container params for box: [center_x, center_y, center_z, half_x, half_y, half_z]
c_params = np.array([container_dims[0]/2, container_dims[1]/2, container_dims[2]/2, container_dims[0]/2, container_dims[1]/2, container_dims[2]/2])
pos, qs, ov = run_sage_engine(target_n=23, skeleton_data=skel, container_type=0, container_params=c_params)

# Export scene
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
    scene.add_geometry(unit_mesh, transform=matrix)

# Add container box (aligned with the pack center)
box = trimesh.creation.box(extents=container_dims)
box.apply_translation(container_dims/2)
box.visual.face_colors = [255, 255, 255, 50] 
scene.add_geometry(box)

scene.export('SAGE_3D_SOLVER/VISUALIZATION_ASSETS/SAGE_Engine_23Unit.glb')
print('Regenerated 23-Unit Pack with correct container dimensions')
