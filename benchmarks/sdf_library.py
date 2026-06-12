import numpy as np
from numba import njit

@njit
def sdf_box(p, extents):
    """SDF for a box at origin with given extents (half-widths)."""
    q = np.abs(p) - extents
    return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

@njit
def sdf_cylinder(p, radius, height):
    """SDF for a Z-aligned cylinder at origin."""
    d = np.abs(np.array([np.sqrt(p[0]**2 + p[1]**2), p[2]])) - np.array([radius, height/2])
    return min(max(d[0], d[1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0))

@njit
def sdf_sphere(p, radius):
    """SDF for a sphere at origin."""
    return np.linalg.norm(p) - radius

@njit
def get_container_sdf(p, container_type, params):
    """
    Returns the distance to the boundary of the container.
    Internal points have negative values (distance to wall).
    Boundary repulsion should trigger when p + radius > 0 (near/outside wall).
    """
    if container_type == 0: # Box
        # params: [center_x, center_y, center_z, half_x, half_y, half_z]
        center = params[0:3]
        extents = params[3:6]
        return sdf_box(p - center, extents)
    elif container_type == 1: # Cylinder
        # params: [center_x, center_y, center_z, radius, height]
        center = params[0:3]
        radius = params[3]
        height = params[4]
        return sdf_cylinder(p - center, radius, height)
    elif container_type == 2: # Sphere
        # params: [center_x, center_y, center_z, radius]
        center = params[0:3]
        radius = params[3]
        return sdf_sphere(p - center, radius)
    return 0.0

@njit
def get_sdf_gradient(p, container_type, params, eps=1e-4):
    """Calculates the normal of the SDF at point p."""
    d = get_container_sdf(p, container_type, params)
    nx = get_container_sdf(p + np.array([eps, 0, 0]), container_type, params) - d
    ny = get_container_sdf(p + np.array([0, eps, 0]), container_type, params) - d
    nz = get_container_sdf(p + np.array([0, 0, eps]), container_type, params) - d
    grad = np.array([nx, ny, nz]) / eps
    norm = np.linalg.norm(grad)
    if norm < 1e-8: return np.array([0.0, 0.0, 1.0])
    return grad / norm
