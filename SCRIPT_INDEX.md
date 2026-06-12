# SAGE3D Technical Index: Experimental Scripts & Algorithms

This document indexes the evolution of the solver, from early physics-based drafts to the final high-performance gradient engine.

---

## 1. Core Mathematical Foundation
*   **`sage_engine.py` (The Differentiable Skeletal Engine):** The state-of-the-art solver. Uses analytical gradients of the Lumelsky distance function for simultaneous 6-DOF optimization. This engine achieved the "Perfect 23" zero-collision pack.
*   **`generate_mesh.py` (Geometry Engine):** Mathematically defines the 1-ton tetrapod using tetrahedral symmetry and cylindrical revolution.

---

## 2. The Verification System
*   **`diagnose_overlaps.py`:** The definitive "Truth" script. Performs a high-precision audit of finished GLB results using the Lumelsky distance kernel to report collision depths in meters.
*   **`test_collision.py` / `repro_collision.py`:** Unit tests and visual debugging tools for the segment-to-segment distance logic.

---

## 3. Evolution of the Optimization Engines

### Phase 1: Physics-Based Dynamics (PBD)
*   **Scripts:** `packer_v35.py`, `packer_massive.py`.
*   **Approach:** Uses vertex-to-boundary constraints and iterative repulsion.
*   **Limitation:** Jammed at 22 units due to lack of complex rotational interlocking logic.

### Phase 2: Sequential Heuristic Search
*   **Scripts:** `packer_v44.py`, `packer_v53.py`.
*   **Approach:** Adds units one-by-one, searching for "local holes."
*   **Limitation:** Capped at 13-15 units because early placements did not account for global packing density requirements.

### Phase 3: Monte Carlo Structured Resolvers
*   **Scripts:** `packer_v55.py`, `packer_v60.py`.
*   **Approach:** Simultaneous SO(3) perturbation with a Metropolis-Hastings schedule (Simulated Annealing).
*   **Success:** Achieved the first "Perfect 22" (0.00m overlap) by navigating geometric barriers via high-temperature "jitter."

### Phase 4: Analytical Gradient Optimization
*   **Scripts:** `sage_engine.py` (v65+).
*   **Approach:** Replaces random search with **Analytical Gradients**. Calculates repulsion forces and torques directly from the Lumelsky distance function.
*   **Success:** Achieved the "Perfect 23" frontier in <30 seconds.

---

## 4. Specialized Research Tools
*   **`packer_v62_variations.py`:** Explores the "Topology Space" to prove that multiple distinct interlocking configurations exist for a 22-unit pack.
*   **`stress_test_limits.py`:** Systematically identifies the "Geometric Jamming" point by increasing units until the solver fails to achieve zero-collision convergence.
*   **`audit_frontier.py`:** Batch-processes multiple export results to generate density statistics for the paper's results section.
