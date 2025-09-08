import numpy as np
from .math import Vec3, Xform, EPS

class Hit:
    __slots__ = ("p","n","t","id","mat")
    def __init__(self):
        self.p = None
        self.n = None
        self.t = np.inf
        self.id = -1
        self.mat = None

class Prim:
    def __init__(self, kind: str, params, xf: Xform | None = None, beta: float = 0.0, pid: int = 0, op: str = "solid"):
        self.kind = kind  # sphere, box, torus, capsule, plane
        self.params = np.asarray(params, dtype=np.float64)
        self.xf = xf or Xform()
        self.beta = float(beta)
        self.pid = int(pid)
        self.op = op  # 'solid' or 'subtract'

# --- Signed distance primitives ---

def sd_sphere(p, r):
    return np.linalg.norm(p) - r

def sd_box(p, b):
    q = np.abs(p) - b
    return np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0)

def sd_capsule(p, a, b, r):
    pa = p - a; ba = b - a
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    return np.linalg.norm(pa - ba * h) - r

def sd_torus(p, R, r):
    qx = np.sqrt(p[0]*p[0] + p[2]*p[2]) - R
    return np.sqrt(qx*qx + p[1]*p[1]) - r

def pia_scale(r: float, beta: float) -> float:
    return r * (1.0 + 0.5 * 0.25 * beta * r * r)

class Scene:
    def __init__(self):
        self.prims: list[Prim] = []
        self.global_beta = 0.0
        self.env_light = 0.25
        self.bg_color = (0.08,0.08,0.1)
    def add(self, prim: Prim):
        self.prims.append(prim); return prim.pid
    def sdf(self, pw):
        d = 1e18; pid = -1; mat = None
        for pr in self.prims:
            M = np.linalg.inv(pr.xf.M)
            pl = (M @ np.array([pw[0], pw[1], pw[2], 1.0]))[:3]
            if pr.kind == 'sphere':
                r = pia_scale(pr.params[3], pr.beta + self.global_beta)
                dl = sd_sphere(pl, r)
            elif pr.kind == 'box':
                dl = sd_box(pl, pr.params[:3])
            elif pr.kind == 'torus':
                R, r = pr.params[0], pr.params[1]
                r = pia_scale(r, pr.beta + self.global_beta)
                dl = sd_torus(pl, R, r)
            else:
                continue
            if pr.op == 'subtract':
                dl = -dl
            if dl < d:
                d = dl; pid = pr.pid; mat = pr
        return d, pid, mat
    def normal(self, p):
        e = 1e-4
        dx = self.sdf((p[0]+e,p[1],p[2]))[0] - self.sdf((p[0]-e,p[1],p[2]))[0]
        dy = self.sdf((p[0],p[1]+e,p[2]))[0] - self.sdf((p[0],p[1]-e,p[2]))[0]
        dz = self.sdf((p[0],p[1],p[2]+e))[0] - self.sdf((p[0],p[1],p[2]-e))[0]
        n = np.array([dx,dy,dz])
        ln = np.linalg.norm(n)
        return n/ln if ln > 1e-12 else np.array([0,1,0])

def ray_march(scene: Scene, ro, rd, tmin=0.0, tmax=100.0, max_steps=256, hit_eps=1e-4):
    t = tmin
    for _ in range(max_steps):
        p = ro + rd * t
        d, pid, mat = scene.sdf(p)
        if d < hit_eps:
            h = Hit(); h.p = p; h.n = scene.normal(p); h.t = t; h.id = pid; h.mat = mat; return h
        t += max(d, 1e-3)
        if t > tmax:
            break
    return None
