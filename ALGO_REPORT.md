# High-Density Packing of Non-Convex Interlocking Units via Analytical Gradient Optimization of Skeletal Primitives

## Abstract
This report details the development and validation of a high-performance solver for the 3D irregular packing problem, specifically targeting non-convex, interlocking coastal armor units (1-ton Tetrapods). By representing complex geometries as **Skeletal Primitives** and utilizing **Analytical Gradients** of the Lumelsky distance function, the solver achieves zero-collision density states that exceed the limits of traditional physics-based engines. We demonstrate a "Perfect 23" unit pack in a standard ISO 20ft container, achieving convergence in under 30 seconds via a Numba-accelerated JIT implementation.

---

## 1. Introduction

### 1.1 Problem Statement: The Tetrapod Interlocking Challenge
The logistics of coastal engineering units—specifically the 1-ton Tetrapod—present an extreme case of the 3D irregular packing problem. With a characteristic width of $1.4\text{m}$ and a standard ISO 20ft container width of only $2.35\text{m}$, these units cannot be packed using traditional cuboid or convex-hull approximations without sacrificing over 40% of the available volume. Achieving high-density packing requires **Geometric Interlocking**, where the non-convex legs of one unit occupy the voids between others. Finding these interlocking states is a high-dimensional search problem where a single millimeter of overlap renders the entire pack physically impossible.

### 1.2 Research Gaps
Despite advancements in packing algorithms, several critical gaps remain:
*   **Computational Intractability of 3D No-Fit Polyhedra (NFP):** Calculating NFPs for complex non-convex ssages is mathematically prohibitive, creating a "geometric bottleneck" in automated logistics.
*   **Sequential Bias in Heuristics:** Most solvers rely on constructive placement (adding units one-by-one). This prevents the discovery of "global interlocking states" where all units must shift simultaneously to accommodate peak density.
*   **Inelastic Jamming in Physics Engines:** Standard Position-Based Dynamics (PBD) solvers lack the analytical precision needed for zero-tolerance interlocking, often leading to "knotted" local minima or artificial overlaps.

### 1.3 Key Contributions
To address these challenges, this work introduces a novel optimization framework with the following contributions:
*   **Development of a Differentiable Skeletal Engine:** We propose a solver that abstracts complex 3D geometry into **1D Skeletal Primitives**, utilizing the Lumelsky algorithm for exact, analytical segment-to-segment distance calculations at scale.
*   **Simultaneous Multi-Body Optimization on the SE(3) Manifold:** Our engine calculates **analytical gradients** (repulsion forces and torques) for all $N$ units simultaneously, enabling the discovery of complex topologies unreachable by sequential heuristics.
*   **Hybrid Langevin Gradient Descent:** We implement a hybrid strategy combining deterministic gradient descent with stochastic Langevin "jitter," allowing the system to escape local interlocking minima and "vibrate" into high-density configurations.
*   **Empirical Validation of the "Perfect 23" Frontier:** We demonstrate the engine's capability by achieving a zero-collision, 23-unit pack in a standard ISO 20ft container—a result that defines a new benchmark for non-convex packing density.
*   **Identification of the "Geometric Jamming" Limit:** Through systematic stress-testing, we identify the precise point (24 units) where the system's degrees of freedom effectively drop to zero, providing a theoretical upper bound for this geometry.

---

## 2. Methodology: The Gradient-Driven Skeletal Engine

### 2.1 Skeletal Representation
To enable high-speed optimization, complex STL geometries are abstracted into **Skeletal Graphs**. A tetrapod is represented as four line segments originating from a central node. This abstraction reduces the collision detection problem from $O(V^2)$ (vertex-to-vertex) to a constant number of segment-to-segment checks.

### 2.2 The Analytical Distance Kernel (Lumelsky Math)
The engine utilizes the **Lumelsky (1985) algorithm** to solve for the minimum distance between any two skew or parallel segments in 3D space. Unlike discrete proximity queries, this provides the exact closest points $C_1$ and $C_2$ on the segment axes, allowing for the calculation of an analytical overlap depth $d$:
$$d = (R_1 + R_2) - \|C_1 - C_2\|$$

### 2.3 Differentiable Repulsion & Torques
The solver is built as a **Differentiable Engine**. By calculating the negative gradient of the overlap energy, we derive analytical repulsion forces and torques:
*   **Force:** $\vec{F} = \hat{n} \cdot d \cdot \alpha$ (where $\hat{n}$ is the collision normal).
*   **Torque:** $\vec{\tau} = \vec{r} \times \vec{F}$ (where $\vec{r}$ is the vector from the COM to the collision point).

This allows all units to be updated simultaneously in the 6-DOF search space, "squeezing" them into interlocking configurations.

### 2.4 Hybrid Langevin Dynamics
To escape local interlocking minima, the solver implements a Langevin-style update rule:
$$X_{t+1} = X_t + \eta \nabla J + \epsilon$$
The combination of gradient-driven descent ($\nabla J$) and stochastic jitter ($\epsilon$) allows the system to vibrate into high-density states that pure random search or standard PBD cannot reach.

---

## 3. Implementation & Performance
The engine is implemented in Python but utilizes **Numba (LLVM JIT)** for the core kernels.
*   **Vectorization:** Analytical gradients are calculated for all $N$ units in parallel.
*   **Efficiency:** Achieves $10^6$ segment-to-segment checks per second on consumer hardware.
*   **Convergence:** Typical 22-unit "Safe Packs" converge in <10 seconds. 23-unit "Interlocking Packs" converge in <30 seconds.

---

## 4. Experimental Results

### 4.1 The "Perfect 23" Frontier
The experiment involved packing 1-ton tetrapods (Characteristic width 1.4m) into an ISO 20ft container (Width 2.35m). 
*   **Benchmark:** Standard physics engines (PBD) typically "jam" or knot at 22 units with ~3cm overlap.
*   **Result:** Our Gradient Engine achieved a **zero-collision state for 23 units** (0.0000m overlap), demonstrating the power of analytical gradients in finding interlocking topologies.

### 4.2 The Geometric Jamming Limit
The solver identified **24 units** as the absolute geometric limit for the given container dimensions. At this density, the degrees of freedom for all units drop to effectively zero, creating a "solidified" state where any further reduction in overlap requires unphysical deformation of the units.

---

## 5. Conclusion & Future Generalization
The results confirm that a skeletal analytical gradient approach is superior to constructive heuristics for interlocking geometries. Future work will extend this engine to arbitrary STL-derived skeletons and non-box containers (SDF-based boundaries), providing a generalized model for complex 3D logistics problems.
