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
    D = np.sqrt(X*X + Y*Y) - re
    band = np.clip(0.5 - D/(span/res*2.0), 0.0, 1.0)
    img = (255*(1.0 - band)).astype(np.uint8)
    return img
