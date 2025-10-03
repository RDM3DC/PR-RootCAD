import numpy as np
from .math import Xform, clamp

# ---- Limits ----
MAX_PRIMS = 48  # keep in sync with shader arrays

# ---- Kinds & Ops ----
KIND_NONE, KIND_SPHERE, KIND_BOX, KIND_CAPSULE, KIND_TORUS, KIND_MOBIUS, KIND_SUPERELLIPSOID, KIND_QUASICRYSTAL, KIND_TORUS4D, KIND_MANDELBULB, KIND_KLEIN, KIND_MENGER, KIND_HYPERBOLIC, KIND_GYROID, KIND_TREFOIL = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
OP_SOLID, OP_SUBTRACT = 0, 1

def pia_scale(r, beta):  # toy metric scaling
    return r * (1.0 + 0.125 * beta * r * r)

# --- Primitive SDFs (local space) ---
def sd_sphere(p, r):
    return np.linalg.norm(p) - r

def sd_box(p, b):  # b = (sx,sy,sz)
    q = np.abs(p) - b
    return np.linalg.norm(np.maximum(q,0.0)) + min(q.max(), 0.0)

def sd_capsule_y(p, r, h):  # Y-axis capsule height=h
    a = np.array([0.0,-0.5*h,0.0])
    b = np.array([0.0, 0.5*h,0.0])
    pa = p - a; ba = b - a
    t = clamp(np.dot(pa,ba)/(np.dot(ba,ba)+1e-12), 0.0, 1.0)
    return np.linalg.norm(pa - ba*t) - r

def sd_torus_y(p, R, r):
    qx = np.sqrt(p[0]*p[0] + p[2]*p[2]) - R
    return np.sqrt(qx*qx + p[1]*p[1]) - r

def sd_torus4d(p, R1, R2, r, w_slice=0.0):
    """4D torus (duocylinder) with 3D cross-section at w=w_slice
    R1, R2 are major radii of two orthogonal circles
    r is minor radius (tube thickness)
    w_slice is the 4th dimension coordinate for cross-section
    """
    # In 4D: x²+y² and z²+w² form two perpendicular circles
    # For 3D visualization, we treat z as the 4th dimension w
    circle1_radius = np.sqrt(p[0]*p[0] + p[1]*p[1])
    circle2_radius = np.sqrt(p[2]*p[2] + w_slice*w_slice)
    
    # Distance to the 4D torus surface
    d1 = circle1_radius - R1
    d2 = circle2_radius - R2
    return np.sqrt(d1*d1 + d2*d2) - r

def sd_mandelbulb(p, power=8.0, bailout=2.0, max_iter=16):
    """3D Mandelbulb fractal distance estimation
    power: fractal power (8.0 is classic)
    bailout: escape radius 
    max_iter: iteration limit for performance
    """
    z = p.copy()
    dr = 1.0
    r = 0.0
    
    for i in range(int(max_iter)):
        r = np.linalg.norm(z)
        if r > bailout:
            break

        # Avoid singularities but keep iterating for inner points
        r_safe = max(r, 1e-6)

        theta = np.arccos(np.clip(z[2] / r_safe, -1.0, 1.0))
        phi = np.arctan2(z[1], z[0])

        zr = r_safe ** (power - 1.0)
        dr = zr * power * dr + 1.0

        zr *= r_safe
        sin_theta = np.sin(theta * power)
        z = zr * np.array([
            sin_theta * np.cos(phi * power),
            sin_theta * np.sin(phi * power),
            np.cos(theta * power)
        ]) + p
    
    # Improved distance estimation
    if r < bailout:
        return 0.0  # Inside the set
    else:
        return 0.5 * np.log(r) * r / max(dr, 1e-6)


def _palette(t, a=None, b=None, c=None, d=None):
    """Simple palette matching the GLSL 'palette' helper used in shaders.
    a,b,c,d are RGB triplets or scalars. Returns np.array RGB in [0,1].
    """
    if a is None: a = np.array([0.50,0.50,0.50], dtype=np.float64)
    if b is None: b = np.array([0.50,0.50,0.50], dtype=np.float64)
    if c is None: c = np.array([1.00,1.00,1.00], dtype=np.float64)
    if d is None: d = np.array([0.00,0.33,0.67], dtype=np.float64)
    # ensure arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    return a + b * np.cos(2.0 * np.pi * (c * t + d))


def mandelbulb_iter_cpu(p, power=8.0, bailout=2.0, max_iter=16, orbit_shell=0.0):
    """CPU replicate of the Mandelbulb orbit calculation used by the shader.
    Returns dict with trapPlane, trapShell, r, nu, dr, lastZ, iter
    """
    z = p.copy().astype(np.float64)
    dr = 1.0
    r = 0.0
    trapPlane = 1e9
    trapShell = 1e9
    lastZ = z.copy()
    iters = 0
    for i in range(int(max_iter)):
        r = np.linalg.norm(z)
        if r > bailout:
            iters = i
            break
        r_safe = max(r, 1e-12)
        trapPlane = min(trapPlane, abs(z[1]))
        if orbit_shell > 0.0:
            trapShell = min(trapShell, abs(r_safe - orbit_shell))
        theta = np.arccos(np.clip(z[2] / r_safe, -1.0, 1.0))
        phi = np.arctan2(z[1], z[0])
        zr = r_safe ** (power - 1.0)
        dr = dr * power * zr + 1.0
        zr = zr * r_safe
        sin_t = np.sin(theta * power)
        newZ = zr * np.array([
            sin_t * np.cos(phi * power),
            sin_t * np.sin(phi * power),
            np.cos(theta * power)
        ], dtype=np.float64)
        z = newZ + p
        lastZ = z.copy()
        iters = i + 1

    rr = max(r, 1e-12)
    log_base = max(power, 1.001)
    if rr > 1.0:
        nu = float(iters) + 1.0 - np.log(np.log(rr)) / np.log(log_base)
    else:
        nu = float(iters)

    return {
        'trapPlane': float(trapPlane),
        'trapShell': float(trapShell),
        'r': float(r),
        'nu': float(nu),
        'dr': float(dr),
        'lastZ': lastZ,
        'iter': int(iters)
    }


def mandelbulb_color_cpu(p, power=8.0, bailout=2.0, max_iter=16, mode=1, ni_scale=0.08, orbit_shell=1.0):
    """Compute a color for point p inside/outside Mandelbulb matching shader modes.
    mode: 0 smooth escape, 1 orbit trap, 2 angular/phase
    """
    ob = mandelbulb_iter_cpu(p, power, bailout, max_iter, orbit_shell)
    # palettes and constants similar to fragment shader
    A = np.array([0.50, 0.50, 0.50], dtype=np.float64)
    B = np.array([0.50, 0.50, 0.50], dtype=np.float64)
    C = np.array([1.00, 1.00, 1.00], dtype=np.float64)
    D = np.array([0.00, 0.33, 0.67], dtype=np.float64)

    if mode == 0:
        t = ob['nu'] * float(ni_scale)
        return np.clip(_palette(t, A, B, C, D), 0.0, 1.0)
    elif mode == 1:
        a = np.exp(-8.0 * ob['trapPlane'])
        b = np.exp(-6.0 * ob['trapShell'])
        t = max(0.0, min(1.0, a + 0.5 * b))
        base = (1.0 - t) * np.array([0.1,0.2,0.5]) + t * np.array([0.9,0.9,0.2])
        ni = (ob['nu'] * max(float(ni_scale) * 1.25, 0.001)) % 1.0
        return np.clip(base * (0.85 + 0.3 * ni), 0.0, 1.0)
    else:
        z = ob['lastZ']
        r_safe = max(np.linalg.norm(z), 1e-12)
        phi = np.arctan2(z[1], z[0])
        theta = np.arccos(np.clip(z[2] / r_safe, -1.0, 1.0))
        h = (phi / (2.0 * np.pi)) % 1.0
        s = np.clip(theta / np.pi, 0.0, 1.0)
        v = 0.9
        c = v * s
        x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
        if h < 1.0/6.0:
            rgb = np.array([c, x, 0.0])
        elif h < 2.0/6.0:
            rgb = np.array([x, c, 0.0])
        elif h < 3.0/6.0:
            rgb = np.array([0.0, c, x])
        elif h < 4.0/6.0:
            rgb = np.array([0.0, x, c])
        elif h < 5.0/6.0:
            rgb = np.array([x, 0.0, c])
        else:
            rgb = np.array([c, 0.0, x])
        rgb = rgb + (v - c)
        accent = (ob['nu'] * max(float(ni_scale) * 1.875, 0.001)) % 1.0
        return np.clip(rgb * (0.85 + 0.25 * accent), 0.0, 1.0)

# --- Primitive SDFs (local space) ---
def sd_sphere(p, r):
    return np.linalg.norm(p) - r

def sd_box(p, b):  # b = (sx,sy,sz)
    q = np.abs(p) - b
    return np.linalg.norm(np.maximum(q,0.0)) + min(q.max(), 0.0)

def sd_capsule_y(p, r, h):  # Y-axis capsule height=h
    a = np.array([0.0,-0.5*h,0.0])
    b = np.array([0.0, 0.5*h,0.0])
    pa = p - a; ba = b - a
    t = clamp(np.dot(pa,ba)/(np.dot(ba,ba)+1e-12), 0.0, 1.0)
    return np.linalg.norm(pa - ba*t) - r

def sd_torus_y(p, R, r):
    qx = np.sqrt(p[0]*p[0] + p[2]*p[2]) - R
    return np.sqrt(qx*qx + p[1]*p[1]) - r

def sd_torus4d(p, R1, R2, r, w_slice=0.0):
    """4D torus (duocylinder) with 3D cross-section at w=w_slice
    R1, R2 are major radii of two orthogonal circles
    r is minor radius (tube thickness)
    w_slice is the 4th dimension coordinate for cross-section
    """
    # In 4D: x²+y² and z²+w² form two perpendicular circles
    # For 3D visualization, we treat z as the 4th dimension w
    circle1_radius = np.sqrt(p[0]*p[0] + p[1]*p[1])
    circle2_radius = np.sqrt(p[2]*p[2] + w_slice*w_slice)
    
    # Distance to the 4D torus surface
    d1 = circle1_radius - R1
    d2 = circle2_radius - R2
    return np.sqrt(d1*d1 + d2*d2) - r

def sd_mandelbulb(p, power=8.0, bailout=2.0, max_iter=16):
    """3D Mandelbulb fractal distance estimation
    power: fractal power (8.0 is classic)
    bailout: escape radius 
    max_iter: iteration limit for performance
    """
    z = p.copy()
    dr = 1.0
    r = 0.0
    
    for i in range(int(max_iter)):
        r = np.linalg.norm(z)
        if r > bailout:
            break

        # Avoid singularities but keep iterating for inner points
        r_safe = max(r, 1e-6)

        theta = np.arccos(np.clip(z[2] / r_safe, -1.0, 1.0))
        phi = np.arctan2(z[1], z[0])

        zr = r_safe ** (power - 1.0)
        dr = zr * power * dr + 1.0

        zr *= r_safe
        sin_theta = np.sin(theta * power)
        z = zr * np.array([
            sin_theta * np.cos(phi * power),
            sin_theta * np.sin(phi * power),
            np.cos(theta * power)
        ]) + p
    
    # Improved distance estimation
    if r < bailout:
        return 0.0  # Inside the set
    else:
        return 0.5 * np.log(r) * r / max(dr, 1e-6)


def _palette(t, a=None, b=None, c=None, d=None):
    """Simple palette matching the GLSL 'palette' helper used in shaders.
    a,b,c,d are RGB triplets or scalars. Returns np.array RGB in [0,1].
    """
    if a is None: a = np.array([0.50,0.50,0.50], dtype=np.float64)
    if b is None: b = np.array([0.50,0.50,0.50], dtype=np.float64)
    if c is None: c = np.array([1.00,1.00,1.00], dtype=np.float64)
    if d is None: d = np.array([0.00,0.33,0.67], dtype=np.float64)
    # ensure arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    return a + b * np.cos(2.0 * np.pi * (c * t + d))


def mandelbulb_iter_cpu(p, power=8.0, bailout=2.0, max_iter=16, orbit_shell=0.0):
    """CPU replicate of the Mandelbulb orbit calculation used by the shader.
    Returns dict with trapPlane, trapShell, r, nu, dr, lastZ, iter
    """
    z = p.copy().astype(np.float64)
    dr = 1.0
    r = 0.0
    trapPlane = 1e9
    trapShell = 1e9
    lastZ = z.copy()
    iters = 0
    for i in range(int(max_iter)):
        r = np.linalg.norm(z)
        if r > bailout:
            iters = i
            break
        r_safe = max(r, 1e-12)
        trapPlane = min(trapPlane, abs(z[1]))
        if orbit_shell > 0.0:
            trapShell = min(trapShell, abs(r_safe - orbit_shell))
        theta = np.arccos(np.clip(z[2] / r_safe, -1.0, 1.0))
        phi = np.arctan2(z[1], z[0])
        zr = r_safe ** (power - 1.0)
        dr = dr * power * zr + 1.0
        zr = zr * r_safe
        sin_t = np.sin(theta * power)
        newZ = zr * np.array([
            sin_t * np.cos(phi * power),
            sin_t * np.sin(phi * power),
            np.cos(theta * power)
        ], dtype=np.float64)
        z = newZ + p
        lastZ = z.copy()
        iters = i + 1

    rr = max(r, 1e-12)
    log_base = max(power, 1.001)
    if rr > 1.0:
        nu = float(iters) + 1.0 - np.log(np.log(rr)) / np.log(log_base)
    else:
        nu = float(iters)

    return {
        'trapPlane': float(trapPlane),
        'trapShell': float(trapShell),
        'r': float(r),
        'nu': float(nu),
        'dr': float(dr),
        'lastZ': lastZ,
        'iter': int(iters)
    }


def mandelbulb_color_cpu(p, power=8.0, bailout=2.0, max_iter=16, mode=1, ni_scale=0.08, orbit_shell=1.0):
    """Compute a color for point p inside/outside Mandelbulb matching shader modes.
    mode: 0 smooth escape, 1 orbit trap, 2 angular/phase
    """
    ob = mandelbulb_iter_cpu(p, power, bailout, max_iter, orbit_shell)
    # palettes and constants similar to fragment shader
    A = np.array([0.50, 0.50, 0.50], dtype=np.float64)
    B = np.array([0.50, 0.50, 0.50], dtype=np.float64)
    C = np.array([1.00, 1.00, 1.00], dtype=np.float64)
    D = np.array([0.00, 0.33, 0.67], dtype=np.float64)

    if mode == 0:
        t = ob['nu'] * float(ni_scale)
        return np.clip(_palette(t, A, B, C, D), 0.0, 1.0)
    elif mode == 1:
        a = np.exp(-8.0 * ob['trapPlane'])
        b = np.exp(-6.0 * ob['trapShell'])
        t = max(0.0, min(1.0, a + 0.5 * b))
        base = (1.0 - t) * np.array([0.1,0.2,0.5]) + t * np.array([0.9,0.9,0.2])
        ni = (ob['nu'] * max(float(ni_scale) * 1.25, 0.001)) % 1.0
        return np.clip(base * (0.85 + 0.3 * ni), 0.0, 1.0)
    else:
        z = ob['lastZ']
        r_safe = max(np.linalg.norm(z), 1e-12)
        phi = np.arctan2(z[1], z[0])
        theta = np.arccos(np.clip(z[2] / r_safe, -1.0, 1.0))
        h = (phi / (2.0 * np.pi)) % 1.0
        s = np.clip(theta / np.pi, 0.0, 1.0)
        v = 0.9
        c = v * s
        x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
        if h < 1.0/6.0:
            rgb = np.array([c, x, 0.0])
        elif h < 2.0/6.0:
            rgb = np.array([x, c, 0.0])
        elif h < 3.0/6.0:
            rgb = np.array([0.0, c, x])
        elif h < 4.0/6.0:
            rgb = np.array([0.0, x, c])
        elif h < 5.0/6.0:
            rgb = np.array([x, 0.0, c])
        else:
            rgb = np.array([c, 0.0, x])
        rgb = rgb + (v - c)
        accent = (ob['nu'] * max(float(ni_scale) * 1.875, 0.001)) % 1.0
        return np.clip(rgb * (0.85 + 0.25 * accent), 0.0, 1.0)

# --- New Primitives ---
def sd_gyroid(p, scale=1.0, tau=0.0, thickness=0.05):
    """Approximate SDF of a gyroid TPMS shell.
    f(x,y,z) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) - tau
    Distance ~ |f|/|grad f|, then offset by thickness to create shell.
    scale: frequency multiplier; tau: iso-level; thickness: half-thickness.
    """
    import math
    x,y,z = (np.asarray(p, dtype=np.float64) * float(scale)).tolist()
    sx,cx = math.sin(x), math.cos(x)
    sy,cy = math.sin(y), math.cos(y)
    sz,cz = math.sin(z), math.cos(z)
    f = sx*cy + sy*cz + sz*cx - float(tau)
    gx = cx*cy - sz*sx
    gy = cz*cy - sx*sy
    gz = cx*cz - sy*sz
    g = math.sqrt(gx*gx + gy*gy + gz*gz) + 1e-6
    sdf = abs(f) / g
    return sdf/ max(1e-6, float(scale)) - float(thickness)

def sd_trefoil_knot(p, scale=1.0, tube=0.1, samples=96):
    """Approximate distance to a trefoil knot tube using sampled closest point.
    Trefoil centerline C(t) = ((2+cos 3t) cos 2t, (2+cos 3t) sin 2t, sin 3t)
    We search along t in [0,2pi) and take min distance.
    scale scales the curve; tube is tube radius.
    """
    import math
    pt = np.asarray(p, dtype=np.float64) / max(1e-6, float(scale))
    two_pi = 2.0*math.pi
    n1 = max(24, int(samples)//2)
    best = 1e18
    t_best = 0.0
    for i in range(n1):
        t = two_pi * (i / max(1, n1))
        c3, s3 = math.cos(3*t), math.sin(3*t)
        c2, s2 = math.cos(2*t), math.sin(2*t)
        r = 2.0 + c3
        q = np.array([r*c2, r*s2, s3], dtype=np.float64)
        d = np.linalg.norm(pt - q)
        if d < best: best = d; t_best = t
    # refine around best with smaller window
    n2 = max(24, int(samples))
    window = two_pi / float(n2)
    for j in range(n2):
        t = (t_best - 0.5*window) + (j / max(1, n2-1)) * window
        # wrap
        if t < 0.0: t += two_pi
        if t >= two_pi: t -= two_pi
        c3, s3 = math.cos(3*t), math.sin(3*t)
        c2, s2 = math.cos(2*t), math.sin(2*t)
        r = 2.0 + c3
        q = np.array([r*c2, r*s2, s3], dtype=np.float64)
        d = np.linalg.norm(pt - q)
        if d < best: best = d
    return best * float(scale) - float(tube)

class Prim:
    def __init__(self, kind, params, xform=None, beta=0.0, pid=0, op="solid", color=(0.8,0.7,0.6)):
        self.kind  = kind
        self.params= np.asarray(params, dtype=np.float64)
        self.xform = xform or Xform()
        self.beta  = float(beta)
        self.pid   = int(pid)
        self.op    = op  # 'solid' | 'subtract'
        self.color = np.asarray(color, dtype=np.float64)
        # Added Euler + non-uniform scale tracking
        self.euler = np.array([0.0,0.0,0.0], dtype=np.float32)  # degrees (rx, ry, rz)
        self.scale = np.array([1.0,1.0,1.0], dtype=np.float32)

    def set_transform(self, pos=None, euler=None, scale=None):
        from .math import rot_x, rot_y, rot_z, scale3
        if pos is None:
            pos = self.xform.M[:3,3]
        if euler is None:
            euler = self.euler
        if scale is None:
            scale = self.scale
        S = scale3(float(scale[0]), float(scale[1]), float(scale[2]))
        Rx = rot_x(float(euler[0])); Ry = rot_y(float(euler[1])); Rz = rot_z(float(euler[2]))
        M = (S @ Rz @ Ry @ Rx).astype(np.float32)
        M[:3,3] = np.array(pos, dtype=np.float32)
        self.xform.M = M
        self.euler[:] = euler
        self.scale[:] = scale
        # Optional: precompute inverse for shader (future optimization)
        try:
            self.xform.M_inv = np.linalg.inv(self.xform.M)
        except Exception:
            self.xform.M_inv = None

class Scene:
    def __init__(self):
        self.prims = []
        self.global_beta = 0.0
        # Added for viewport compatibility
        self.bg_color = np.array([0.06,0.07,0.10], dtype=np.float32)
        # Using this as a light direction/tint; shader normalizes it
        self.env_light = np.array([0.7,1.0,0.4], dtype=np.float32)
        self._listeners: list = []

    def on_changed(self, cb):
        if callable(cb):
            self._listeners.append(cb)

    def _notify(self):
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass

    def add(self, prim):
        self.prims.append(prim)
        self._notify()
        return prim.pid

    def clear(self):
        self.prims.clear(); self._notify()

    def remove_index(self, idx:int):
        if 0 <= idx < len(self.prims):
            self.prims.pop(idx)
            self._notify()

    def sdf(self, pw):  # CPU fold (union + subtract)
        d = 1e18
        for pr in self.prims:
            Mi = np.linalg.inv(pr.xform.M)
            pl = (Mi @ np.array([pw[0], pw[1], pw[2], 1.0]))[:3]
            if pr.kind in (KIND_SPHERE, 'sphere'):
                r = pia_scale(pr.params[0], pr.beta + self.global_beta)
                di = sd_sphere(pl, r)
            elif pr.kind in (KIND_BOX, 'box'):
                di = sd_box(pl, pr.params[:3])
            elif pr.kind in (KIND_CAPSULE, 'capsule'):
                r = pia_scale(pr.params[0], pr.beta + self.global_beta)
                di = sd_capsule_y(pl, r, pr.params[1])
            elif pr.kind in (KIND_TORUS, 'torus'):
                R, r0 = pr.params[0], pr.params[1]
                r = pia_scale(r0, pr.beta + self.global_beta)
                di = sd_torus_y(pl, R, r)
            elif pr.kind in (KIND_MOBIUS, 'mobius'):
                R, w = pr.params[0], pr.params[1]
                di = sd_mobius(pl, float(R), float(w), samples=64)
            elif pr.kind in (KIND_SUPERELLIPSOID, 'superellipsoid'):
                r, power = pr.params[0], max(1.0, pr.params[1])
                # Allow beta to modulate soft rounding slightly
                di = sd_superellipsoid(pl, r, power)
            elif pr.kind in (KIND_QUASICRYSTAL, 'quasicrystal'):
                # Params: [scale, iso, thickness, 0]
                sc, iso, th = float(pr.params[0]), float(pr.params[1]), float(max(1e-3, pr.params[2]))
                val, K = qc_value_and_bound(pl, sc)
                di = (abs(val - iso) / max(K,1e-6)) - th
            elif pr.kind in (KIND_TORUS4D, 'torus4d'):
                # Params: [R1, R2, r, w_slice] - two major radii, minor radius, 4D slice position
                R1, R2, r = pr.params[0], pr.params[1], pr.params[2]
                w_slice = pr.params[3] if len(pr.params) > 3 else 0.0
                di = sd_torus4d(pl, R1, R2, r, w_slice)
            elif pr.kind in (KIND_MANDELBULB, 'mandelbulb'):
                # Params: [power, bailout, max_iter, scale]
                power = max(2.0, pr.params[0]) if len(pr.params) > 0 else 8.0
                bailout = max(1.0, pr.params[1]) if len(pr.params) > 1 else 2.0
                max_iter = int(max(4, pr.params[2])) if len(pr.params) > 2 else 16
                scale = pr.params[3] if len(pr.params) > 3 else 1.0
                di = sd_mandelbulb(pl * scale, power, bailout, max_iter) / scale
            elif pr.kind in (KIND_KLEIN, 'klein'):
                # Params: [scale, n, t_offset, thickness]
                scale = pr.params[0] if len(pr.params) > 0 else 1.0
                n = pr.params[1] if len(pr.params) > 1 else 2.0
                t_offset = pr.params[2] if len(pr.params) > 2 else 0.0
                di = sd_klein_bottle(pl, scale, n, t_offset)
            elif pr.kind in (KIND_MENGER, 'menger'):
                # Params: [iterations, size, 0, 0]
                iterations = max(1, pr.params[0]) if len(pr.params) > 0 else 3
                size = max(0.1, pr.params[1]) if len(pr.params) > 1 else 1.0
                di = sd_menger_sponge(pl, iterations, size)
            elif pr.kind in (KIND_HYPERBOLIC, 'hyperbolic'):
                # Params: [scale, order, symmetry, 0]
                scale = max(0.1, pr.params[0]) if len(pr.params) > 0 else 1.0
                order = max(3, pr.params[1]) if len(pr.params) > 1 else 7
                symmetry = max(3, pr.params[2]) if len(pr.params) > 2 else 3
                di = sd_hyperbolic_tiling(pl, scale, order, symmetry)
            elif pr.kind in (KIND_GYROID, 'gyroid'):
                # Params: [scale, tau, thickness, 0]
                scale = pr.params[0] if len(pr.params) > 0 else 1.0
                tau = pr.params[1] if len(pr.params) > 1 else 0.0
                thickness = pr.params[2] if len(pr.params) > 2 else 0.05
                di = sd_gyroid(pl, scale, tau, thickness)
            elif pr.kind in (KIND_TREFOIL, 'trefoil'):
                # Params: [scale, tube, samples, 0]
                scale = pr.params[0] if len(pr.params) > 0 else 1.0
                tube  = pr.params[1] if len(pr.params) > 1 else 0.1
                samples = int(pr.params[2]) if len(pr.params) > 2 else 96
                di = sd_trefoil_knot(pl, scale, tube, samples)
            else:
                continue
            if pr.op == 'subtract':
                di = -di
                d = max(d, di)
            else:
                d = min(d, di)
        return d, -1, None

    # ---------- GPU packing ----------
    def to_gpu_structs(self, max_prims=MAX_PRIMS):
        n = min(len(self.prims), max_prims)
        kind  = np.zeros(max_prims, dtype=np.int32)
        op    = np.zeros(max_prims, dtype=np.int32)
        beta  = np.zeros(max_prims, dtype=np.float32)
        color = np.zeros((max_prims,3), dtype=np.float32)
        params= np.zeros((max_prims,4), dtype=np.float32)
        xform = np.zeros((max_prims,16), dtype=np.float32)
        xform_inv = np.zeros((max_prims,16), dtype=np.float32)
        for i, pr in enumerate(self.prims[:n]):
            if pr.kind in (KIND_SPHERE, 'sphere'):
                kind[i]  = KIND_SPHERE
                params[i]= [pr.params[0], 0, 0, 0]
            elif pr.kind in (KIND_BOX, 'box'):
                kind[i]  = KIND_BOX
                params[i]= [pr.params[0], pr.params[1], pr.params[2], 0]
            elif pr.kind in (KIND_CAPSULE, 'capsule'):
                kind[i]  = KIND_CAPSULE
                params[i]= [pr.params[0], pr.params[1], 0, 0]
            elif pr.kind in (KIND_TORUS, 'torus'):
                kind[i]  = KIND_TORUS
                params[i]= [pr.params[0], pr.params[1], 0, 0]
            elif pr.kind in (KIND_MOBIUS, 'mobius'):
                kind[i]  = KIND_MOBIUS
                params[i]= [pr.params[0], pr.params[1], 0, 0]
            elif pr.kind in (KIND_SUPERELLIPSOID, 'superellipsoid'):
                kind[i]  = KIND_SUPERELLIPSOID
                params[i]= [pr.params[0], pr.params[1], 0, 0]
            elif pr.kind in (KIND_QUASICRYSTAL, 'quasicrystal'):
                kind[i]  = KIND_QUASICRYSTAL
                params[i]= [pr.params[0], pr.params[1], pr.params[2], 0]
            elif pr.kind in (KIND_TORUS4D, 'torus4d'):
                kind[i]  = KIND_TORUS4D
                # Pack R1, R2, r, w_slice
                w_slice = pr.params[3] if len(pr.params) > 3 else 0.0
                params[i]= [pr.params[0], pr.params[1], pr.params[2], w_slice]
            elif pr.kind in (KIND_MANDELBULB, 'mandelbulb'):
                kind[i]  = KIND_MANDELBULB
                # Pack power, bailout, max_iter, scale
                power = max(2.0, pr.params[0]) if len(pr.params) > 0 else 8.0
                bailout = max(1.0, pr.params[1]) if len(pr.params) > 1 else 2.0
                max_iter = max(4, pr.params[2]) if len(pr.params) > 2 else 16
                scale = pr.params[3] if len(pr.params) > 3 else 1.0
                params[i]= [power, bailout, max_iter, scale]
            elif pr.kind in (KIND_KLEIN, 'klein'):
                kind[i]  = KIND_KLEIN
                # Pack scale, n, t_offset, thickness
                scale = pr.params[0] if len(pr.params) > 0 else 1.0
                n = pr.params[1] if len(pr.params) > 1 else 2.0
                t_offset = pr.params[2] if len(pr.params) > 2 else 0.0
                thickness = pr.params[3] if len(pr.params) > 3 else 0.1
                params[i]= [scale, n, t_offset, thickness]
            elif pr.kind in (KIND_MENGER, 'menger'):
                kind[i]  = KIND_MENGER
                # Pack iterations, size, 0, 0
                iterations = max(1, pr.params[0]) if len(pr.params) > 0 else 3
                size = max(0.1, pr.params[1]) if len(pr.params) > 1 else 1.0
                params[i]= [iterations, size, 0, 0]
            elif pr.kind in (KIND_HYPERBOLIC, 'hyperbolic'):
                kind[i]  = KIND_HYPERBOLIC
                # Pack scale, order, symmetry, 0
                scale = max(0.1, pr.params[0]) if len(pr.params) > 0 else 1.0
                order = max(3, pr.params[1]) if len(pr.params) > 1 else 7
                symmetry = max(3, pr.params[2]) if len(pr.params) > 2 else 3
                params[i]= [scale, order, symmetry, 0]
            elif pr.kind in (KIND_GYROID, 'gyroid'):
                kind[i] = KIND_GYROID
                # Pack scale, tau, thickness, 0
                scale = pr.params[0] if len(pr.params) > 0 else 1.0
                tau = pr.params[1] if len(pr.params) > 1 else 0.0
                thickness = pr.params[2] if len(pr.params) > 2 else 0.05
                params[i] = [scale, tau, thickness, 0]
            elif pr.kind in (KIND_TREFOIL, 'trefoil'):
                kind[i] = KIND_TREFOIL
                # Pack scale, tube, samples, 0 (samples as float)
                scale = pr.params[0] if len(pr.params) > 0 else 1.0
                tube  = pr.params[1] if len(pr.params) > 1 else 0.1
                samples = pr.params[2] if len(pr.params) > 2 else 96
                params[i] = [scale, tube, samples, 0]
            else:
                kind[i]  = KIND_NONE

            op[i]     = OP_SUBTRACT if pr.op == 'subtract' else OP_SOLID
            beta[i]   = np.float32(pr.beta + self.global_beta)
            color[i]  = pr.color[:3].astype(np.float32)
            # Pack transforms column-major for GLSL compatibility
            fwd = pr.xform.M.astype(np.float32)
            try:
                inv = np.linalg.inv(fwd)
            except Exception:
                inv = np.eye(4, dtype=np.float32)
            xform[i]     = fwd.flatten(order="F")
            xform_inv[i] = inv.flatten(order="F")

        return {
            "count": np.int32(n),
            "kind": kind, "op": op, "beta": beta,
            "color": color, "params": params,
            "xform": xform, "xform_inv": xform_inv,
            "bg": self.bg_color.astype(np.float32),
            "env": self.env_light.astype(np.float32)
        }

