# SAGE Developer & Agent Collaboration Guide

Welcome to the **SAGE (Skeletal Analytical Gradient Engine)** repository. This document serves as the shared instructions, code quality standards, and workflow protocol for all human developers and AI agents working on this project. 

All agents **must** read and adhere to this guide before creating files or modifying code.

---

## 📌 1. Naming & Branding Guidelines
*   **Primary Name:** SAGE (Skeletal Analytical Gradient Engine / Differentiable Skeletal Solver).
*   **Rule:** The old name (`hape` / `hape3d`) has been deprecated. Never use these terms in code, variables, filenames, or documentation.
*   **Key Files:**
    *   Core Engine: `engine/sage_engine.py`
    *   3D Visualizer: `visualizer/index.html` (served via `serve.py`)

---

## 📂 2. Repository Architecture
Ensure all files are placed in their respective modular folders:
*   `/engine`: Core mathematical, JIT-accelerated, and geometrical code (e.g., `sage_engine.py`, `extract_skeleton.py`, `sdf_library.py`).
*   `/benchmarks`: Test scripts and theoretical verification datasets (e.g., Williams-Philipse packing calculations).
*   `/results`: GLB/GLTF simulation outputs, divided strictly by research categories (e.g., `/04_Frontier_Research`).
*   `/scripts`: Utilities for post-processing and geometry regeneration.
*   `/visualizer`: Web-based three.js rendering code.
*   `serve.py`: Main dashboard local server.

---

## ⚙️ 3. Code Standards & Numba Compatibility
*   **Numba JIT Acceleration:** The core solver loops in `sage_engine.py` are JIT-compiled using Numba (`@njit`). Do not introduce non-compilable Python types, external object calls, or unsupported library operations inside `@njit` decorated functions.
*   **JSON Serialization Rule:** When exporting coordinate structures or metadata (such as in skeletonization), **always convert NumPy numerical types to native Python types** before saving to JSON:
    *   Use `int(np_val)` instead of `np.int64`.
    *   Use `float(np_val)` instead of `np.float64`.
    *   Use `arr.tolist()` for NumPy arrays.
    *   *Failure to do this will throw serialization exceptions during data exports.*

---

## 🌿 4. Git Branching & Parallel Collaboration Protocol
To work in parallel without causing file lockouts or merge conflicts, adhere to this protocol:

1.  **Work on a Branch:** Never push code directly to the `main` branch. Always create a task-specific branch:
    *   Format: `feature/<agent-task-name>` (e.g., `feature/visualizer-overlay`).
2.  **No Shared Working Directories:** Run parallel sessions in isolated workspace branches (using SAGE's `branch` or `share` workspace setups) to prevent write-collisions on disk.
3.  **Run Pre-Commit Verification:** Before committing your branch, run the core engine locally to ensure Numba compiles and outputs zero-overlap states correctly.
4.  **Open a Pull Request:** Once complete, push your branch and open a PR for review.

---

## 🗺️ 5. Active Task Roadmap
Before starting a task, check this roadmap to ensure no other agent is working on it. Update the status of your assigned task as you proceed.

| Task Description | Assigned Agent | Branch Name | Status |
| :--- | :---: | :---: | :---: |
| **SAGE Visualizer UI Upgrades** (Skeletal toggle overlay, lighting adjustments) | *Unassigned* | `feature/visualizer-upgrades` | 🟥 Planned |
| **Williams-Philipse Spherocylinder Benchmark** (Aspect ratio simulations) | *Unassigned* | `feature/wp-benchmark` | 🟥 Planned |
| **SDF Boundary Generalization** (Integrate arbitrary non-box mesh holds) | *Unassigned* | `feature/sdf-boundaries` | 🟥 Planned |
| **Auto-Skeletonization Pipeline Enhancements** | *Unassigned* | `feature/auto-skeletonize` | 🟥 Planned |
