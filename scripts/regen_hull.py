import sys
sys.path.append('SAGE_3D_SOLVER/engine')
from sage_engine import run_sage_engine
import trimesh
import numpy as np
import json

# Load tetrapod skeleton for segment data
with open('SAGE_3D_SOLVER/assets/extracted_skeleton.json', 'r') as f:
    skel = json.load(f)

# Run engine
# Hull container params: [2.949, 1.176, 1.1965, 1.1, 5.8]
c_params = np.array([2.949, 1.176, 1.1965, 1.1, 5.8], dtype=np.float64)
pos, qs, ov = run_sage_engine(target_n=23, skeleton_data=skel, container_type=1, container_params=c_params)

# Export scene
scene = trimesh.Scene()
tetrapod = trimesh.load('SAGE_3D_SOLVER/assets/tetrapod_1ton.stl')
for i in range(len(pos)):
    matrix = trimesh.transformations.quaternion_matrix(qs[i])
    matrix[:3, 3] = pos[i]
    scene.add_geometry(tetrapod, transform=matrix)

scene.export('SAGE_3D_SOLVER/VISUALIZATION_ASSETS/Generalized_Pack_23.glb')
print('Regenerated Hull Pack')
