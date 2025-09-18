import numpy as np
from .math import Xform, clamp

# ---- Limits ----
MAX_PRIMS = 48  # keep in sync with shader arrays

# ---- Kinds & Ops ----
KIND_NONE, KIND_SPHERE, KIND_BOX, KIND_CAPSULE, KIND_TORUS, KIND_MOBIUS, KIND_SUPERELLIPSOID, KIND_QUASICRYSTAL = 0, 1, 2, 3, 4, 5, 6, 7
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


def sd_mobius(p, R, w, samples=64):
    """Approximate distance to a Möbius strip centered on origin.

    Parameterization S(u,v) = A(u) + v*B(u) where
      A(u) = (R*cos(u), R*sin(u), 0)
      B(u) = (cos(u/2)*cos(u), cos(u/2)*sin(u), sin(u/2))

    For each sampled u, v* = dot(p-A, B)/dot(B,B) (clamped to [-w/2,w/2]).
    Then distance = min over u of |p - (A + v*B)|.
    """
    import math
    best = 1e18
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    for i in range(samples):
        u = 2.0 * math.pi * i / samples
        c = math.cos(u); s = math.sin(u)
        c2 = math.cos(u * 0.5); s2 = math.sin(u * 0.5)
        Ax = R * c; Ay = R * s; Az = 0.0
        Bx = c2 * c; By = c2 * s; Bz = s2
        vx = px - Ax; vy = py - Ay; vz = pz - Az
        bb = Bx*Bx + By*By + Bz*Bz + 1e-12
        vstar = (vx*Bx + vy*By + vz*Bz) / bb
        # clamp
        halfw = 0.5 * w
        if vstar < -halfw: vstar = -halfw
        if vstar > halfw: vstar = halfw
        sx = Ax + vstar * Bx
        sy = Ay + vstar * By
        sz = Az + vstar * Bz
        dx = px - sx; dy = py - sy; dz = pz - sz
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d < best: best = d
    return best

def sd_superellipsoid(p, r, power):
    """Approximate SDF for superellipsoid using L_p norm.

    r: scalar base radius; anisotropy handled by external scale in transform.
    power: p >= 1. Larger p -> boxier.
    We approximate distance as (||p||_p / r - 1) * r.
    """
    px, py, pz = abs(float(p[0])), abs(float(p[1])), abs(float(p[2]))
    pwr = max(1.0, float(power))
    rp = max(1e-6, float(r))
    val = (px**pwr + py**pwr + pz**pwr) ** (1.0/pwr)
    return (val / rp - 1.0) * rp

def qc_value_and_bound(p, scale, dirs=None):
    """Compute quasi-crystal value f and a conservative gradient bound K.

    f(x) = sum_i cos(dot(k_i, x*scale)); with |grad f| <= sum_i |k_i|*scale
    We choose |k_i|=1 so K = n*scale.
    """
    import math
    if dirs is None:
        # 7 directions using golden angle spread
        phis = [i*2.39996322973 for i in range(7)]  # golden angle
        dirs = [
            np.array([math.cos(a)*math.sin(b), math.sin(a)*math.sin(b), math.cos(b)], dtype=np.float64)
            for a,b in [(0.0,0.0)]
        ]
        dirs = []
        for i in range(7):
            a = phis[i]
            z = 1.0 - 2.0*(i+0.5)/7.0
            rxy = math.sqrt(max(1e-6, 1.0 - z*z))
            dirs.append(np.array([math.cos(a)*rxy, math.sin(a)*rxy, z], dtype=np.float64))
    ps = float(scale)
    f = 0.0
    for k in dirs:
        f += math.cos(ps * float(np.dot(k, p)))
    K = len(dirs) * abs(ps)
    return f, max(K, 1e-3)

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
        try:
            import logging
            logging.getLogger("adaptivecad.gui").debug(
                f"Prim transform updated kind={self.kind} pos={pos} euler={euler} scale={scale}\nM=\n{M}")
        except Exception:
            pass
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
            else:
                kind[i]  = KIND_NONE

            op[i]     = OP_SUBTRACT if pr.op == 'subtract' else OP_SOLID
            beta[i]   = np.float32(pr.beta + self.global_beta)
            color[i]  = pr.color[:3].astype(np.float32)
            # Forward (column-major) and inverse for GPU (avoid per-fragment inverse())
            fwd = pr.xform.M.astype(np.float32)
            inv = None
            try:
                inv = np.linalg.inv(fwd)
            except Exception:
                inv = np.eye(4, dtype=np.float32)
            xform[i]     = fwd.T.reshape(16)
            xform_inv[i] = inv.T.reshape(16)

        return {
            "count": np.int32(n),
            "kind": kind, "op": op, "beta": beta,
            "color": color, "params": params,
            "xform": xform, "xform_inv": xform_inv,
            "bg": self.bg_color.astype(np.float32),
            "env": self.env_light.astype(np.float32)
        }

