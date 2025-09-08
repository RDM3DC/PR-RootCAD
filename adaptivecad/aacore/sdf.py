import numpy as np
from .math import Xform, clamp

# ---- Limits ----
MAX_PRIMS = 48  # keep in sync with shader arrays

# ---- Kinds & Ops ----
KIND_NONE, KIND_SPHERE, KIND_BOX, KIND_CAPSULE, KIND_TORUS = 0, 1, 2, 3, 4
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

class Prim:
    def __init__(self, kind, params, xform=None, beta=0.0, pid=0, op="solid", color=(0.8,0.7,0.6)):
        self.kind  = kind
        self.params= np.asarray(params, dtype=np.float64)
        self.xform = xform or Xform()
        self.beta  = float(beta)
        self.pid   = int(pid)
        self.op    = op  # 'solid' | 'subtract'
        self.color = np.asarray(color, dtype=np.float64)

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
            else:
                kind[i]  = KIND_NONE

            op[i]     = OP_SUBTRACT if pr.op == 'subtract' else OP_SOLID
            beta[i]   = np.float32(pr.beta + self.global_beta)
            color[i]  = pr.color[:3].astype(np.float32)
            xform[i]  = pr.xform.M.astype(np.float32).reshape(16)

        return {
            "count": np.int32(n),
            "kind": kind, "op": op, "beta": beta,
            "color": color, "params": params, "xform": xform,
            "bg": self.bg_color.astype(np.float32),
            "env": self.env_light.astype(np.float32)
        }

