# adaptivecad/analytic/scene.py
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PiaModel:
    beta: float = 0.0
    kappa_scale: float = 0.25  # tweak
    def radius(self, r: float) -> float:
        kappa = self.kappa_scale * self.beta
        return r * (1.0 + 0.5 * kappa * r * r)

@dataclass
class Transform:
    T: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    def as_mat(self): return self.T.astype(np.float32)

@dataclass
class Primitive:
    kind: str                 # 'sphere','capsule','torus'
    params: np.ndarray        # vec4 (e.g., sphere: (rx,ry,rz, r); capsule: (ax), etc.)
    color: np.ndarray         # rgb in [0,1]
    pia_beta: float = 0.0
    xform: Transform = field(default_factory=Transform)
    object_id: int = 1

@dataclass
class CSGNode:
    op: str                   # 'union','inter','diff'
    left: int                 # index to primitive or CSG node (negative for CSG node)
    right: int

@dataclass
class Scene:
    # Flat arrays for GPU upload; CSG kept small in MVP
    primitives: list = field(default_factory=list)     # list[Primitive]
    csg_nodes: list = field(default_factory=list)       # list[CSGNode]
    bg_color: tuple = (0.08,0.08,0.1)
    env_light: float = 0.25

    def to_gpu_structs(self):
        # Pack primitives into uniform arrays (MVP: up to 32)
        maxP = 32
        n = min(len(self.primitives), maxP)
        kinds = np.zeros((maxP,), np.int32)
        params = np.zeros((maxP,4), np.float32)
        colors = np.zeros((maxP,3), np.float32)
        betas  = np.zeros((maxP,), np.float32)
        ids    = np.zeros((maxP,), np.int32)
        xforms = np.zeros((maxP,4,4), np.float32)
        for i,p in enumerate(self.primitives[:maxP]):
            kinds[i] = {'sphere':0,'capsule':1,'torus':2}.get(p.kind,0)
            params[i,:] = p.params
            colors[i,:] = p.color
            betas[i] = p.pia_beta
            ids[i] = p.object_id
            xforms[i,:,:] = p.xform.as_mat()
        return dict(n=n, kinds=kinds, params=params, colors=colors, betas=betas, ids=ids, xforms=xforms)
