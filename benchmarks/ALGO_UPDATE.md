# SAGE 3D v2.0: Generalized Skeletal Engine & SDF Boundaries

This update transforms SAGE 3D into a universal 3D packing optimizer.

## New Features

### 1. Arbitrary STL Digestibility
- **Script:** `extract_skeleton.py`
- **Function:** Converts any STL mesh into a "digestible" skeletal JSON format.
- **Process:** Voxelizes the mesh, calculates the Medial Axis (Distance Transform), and simplifies the ridges into primary segments and radii.
- **Usage:** `python extract_skeleton.py` (Edit `stl_file` in `main()` for your custom mesh).

### 2. Modifiable Container Shapes (SDF)
- **Library:** `sdf_library.py`
- **Supported Shapes:** Box, Cylinder, Sphere.
- **Mechanism:** The engine now uses Signed Distance Fields to calculate analytical repulsion gradients from any boundary shape. This allows for packing into silos, drums, or custom hulls.

### 3. Stability & Gravity Audit
- **Feature:** Integrated "Phase 3" in `sage_engine.py`.
- **Function:** Applies a constant downward force (gravity) to the finalized pack and measures the **RMSD** (Root Mean Square Deviation).
- **Validation:** A low RMSD (e.g., < 0.05m) proves that the units are physically interlocked and stable, rather than just floating in a collision-free state.

## How to Run
1.  **Extract Skeleton:** `python sage_tetrapod/extract_skeleton.py` (Generates `extracted_skeleton.json`).
2.  **Optimize Pack:** `python sage_tetrapod/sage_engine.py` (Uses the JSON and a selected SDF container).
3.  **View Results:** GLB files are saved in `export_results/05_Generalized_Tests/`.

---
*Developed by Gemini CLI Agent*
