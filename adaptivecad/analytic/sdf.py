import numpy as np


def pia_radius(r: float, beta: float) -> float:
    # toy πₐ: r_eff = r * (1 + 0.5 * κ r^2), κ ∝ β
    kappa = 0.25 * beta
    return float(r * (1.0 + 0.5 * kappa * r * r))


def raster_sphere_cross_section(r: float, beta: float, res: int = 512):
    re = pia_radius(r, beta)
    span = 1.6 * re
    xs = np.linspace(-span, span, res)
    ys = np.linspace(-span, span, res)
    X, Y = np.meshgrid(xs, ys)
    D = np.sqrt(X * X + Y * Y) - re
    band = np.clip(0.5 - D / (span / res * 2.0), 0.0, 1.0)
    img = (255 * (1.0 - band)).astype(np.uint8)
    return img


# --- Signed Distance Functions (SDFs) ---


def vec_length(v):
    """Compute the length of a vector."""
    return np.sqrt(np.sum(np.array(v) ** 2))


def vec_subtract(a, b):
    """Subtract vector b from vector a."""
    return tuple(aa - bb for aa, bb in zip(a, b))


def vec_add(a, b):
    """Add vector b to vector a."""
    return tuple(aa + bb for aa, bb in zip(a, b))


def vec_dot(a, b):
    """Compute the dot product of vectors a and b."""
    return sum(aa * bb for aa, bb in zip(a, b))


def vec_scale(v, s):
    """Scale a vector by a scalar."""
    return tuple(vv * s for vv in v)


def sdf_sphere(point, center, radius):
    """SDF for a sphere."""
    return vec_length(vec_subtract(point, center)) - radius


def sdf_box(point, center, size):
    """SDF for a box."""
    p = vec_subtract(point, center)

    # Adjust for πₐ geometry if needed (for now using Euclidean)
    q = (abs(p[0]) - size[0] / 2, abs(p[1]) - size[1] / 2, abs(p[2]) - size[2] / 2)

    # Inside distance (negative)
    inside_dist = min(max(q[0], max(q[1], q[2])), 0.0)

    # Outside distance (positive)
    outside_dist = vec_length((max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)))

    return inside_dist + outside_dist


def sdf_cylinder(point, center, radius, height):
    """SDF for a cylinder."""
    p = vec_subtract(point, center)

    # Distance in XZ plane (circular base)
    d_xz = vec_length((p[0], p[2])) - radius

    # Distance in Y direction (height)
    d_y = abs(p[1]) - height / 2

    # Combine distances
    inside_dist = min(max(d_xz, d_y), 0.0)
    outside_dist = vec_length((max(d_xz, 0.0), max(d_y, 0.0)))

    return inside_dist + outside_dist


def sdf_capsule(point, center, radius, height):
    """SDF for a capsule (cylinder with hemispheres at the ends)."""
    p = vec_subtract(point, center)

    # Clamp point to the cylinder's line segment
    y = max(min(p[1], height / 2), -height / 2)

    # Vector from nearest point on line segment to the point
    v = (p[0], p[1] - y, p[2])

    return vec_length(v) - radius


def sdf_torus(point, center, major_radius, minor_radius):
    """SDF for a torus."""
    p = vec_subtract(point, center)

    # Distance to the circle defining the torus's centerline
    q = (vec_length((p[0], p[2])) - major_radius, p[1])

    # Distance from point to the tube
    return vec_length(q) - minor_radius
