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
    kind: str                 # 'sphere','box','cylinder','capsule','torus'
    params: np.ndarray        # vec4 (e.g., sphere: (rx,ry,rz, r); capsule: (ax), etc.)
    color: np.ndarray         # rgb in [0,1]
    pia_beta: float = 0.0
    xform: Transform = field(default_factory=Transform)
    object_id: int = 1
    
    @classmethod
    def from_analytic_primitive(cls, primitive):
        """Convert from an AnalyticPrimitive to a Scene Primitive."""
        kind = type(primitive).__name__.lower().replace('analytic', '')
        
        # Extract parameters based on primitive type
        if hasattr(primitive, 'radius'):
            if hasattr(primitive, 'height'):  # Cylinder or Capsule
                params = np.array([*primitive.position, primitive.radius, primitive.height], dtype=np.float32)
            elif hasattr(primitive, 'minor_radius'):  # Torus
                params = np.array([*primitive.position, primitive.major_radius, primitive.minor_radius], dtype=np.float32)
            else:  # Sphere
                params = np.array([*primitive.position, primitive.radius, 0], dtype=np.float32)
        elif hasattr(primitive, 'size'):  # Box
            params = np.array([*primitive.position, *primitive.size], dtype=np.float32)
        else:
            # Default parameters
            params = np.array([*primitive.position, 1.0, 0], dtype=np.float32)
        
        # Extract color
        color = np.array(primitive.color, dtype=np.float32)
        
        # Create and return the Primitive
        return cls(
            kind=kind,
            params=params,
            color=color,
            pia_beta=getattr(primitive, 'beta', 0.0),
            object_id=getattr(primitive, 'id', 1)
        )

@dataclass
class CSGNode:
    op: str                   # 'union','inter','diff'
    left: int                 # index to primitive or CSG node (negative for CSG node)
    right: int

@dataclass
class Scene:
    # Flat arrays for GPU upload; CSG kept small in MVP
    primitives: list = field(default_factory=list)     # list[Primitive]
    csg_nodes: list = field(default_factory=list)      # list[CSGNode]
    bg_color: tuple = (0.08, 0.08, 0.1)
    env_light: float = 0.25
    pia_global_beta: float = 0.0

    def add_primitive(self, primitive):
        """Add a primitive to the scene."""
        if hasattr(primitive, 'to_scene_primitive'):
            # If it's an AnalyticPrimitive, convert it
            scene_prim = primitive
        else:
            # Otherwise create a new primitive from the object
            scene_prim = Primitive.from_analytic_primitive(primitive)
            
        self.primitives.append(scene_prim)
        return len(self.primitives) - 1  # Return index of added primitive

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
        
        kind_map = {
            'sphere': 0,
            'box': 1,
            'cylinder': 2,
            'capsule': 3,
            'torus': 4
        }
        
        for i, p in enumerate(self.primitives[:maxP]):
            kinds[i] = kind_map.get(p.kind, 0)
            params[i,:] = p.params
            colors[i,:] = p.color
            betas[i] = p.pia_beta if p.pia_beta > 0 else self.pia_global_beta
            ids[i] = getattr(p, 'object_id', 1) % 2147483647  # Ensure it fits in a 32-bit integer
            xforms[i,:,:] = p.xform.as_mat()
            
        return dict(n=n, kinds=kinds, params=params, colors=colors, betas=betas, ids=ids, xforms=xforms)
