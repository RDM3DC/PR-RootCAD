import math
import numpy as np

EPS = 1e-8
REL_EPS = 1e-8

def feq(a: float, b: float, tol: float = REL_EPS) -> bool:
    return abs(a - b) <= tol * max(1.0, max(abs(a), abs(b)))

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

class Vec3(np.ndarray):
    @staticmethod
    def of(x: float, y: float, z: float) -> 'Vec3':
        return np.asarray([x, y, z], dtype=np.float64).view(Vec3)
    def norm(self) -> float:
        return float(np.linalg.norm(self))
    def unit(self) -> 'Vec3':
        n = self.norm()
        return self if n < EPS else Vec3.of(*(self / n))

class Xform:
    def __init__(self, M: np.ndarray | None = None):
        self.M = np.eye(4, dtype=np.float64) if M is None else M
    @staticmethod
    def identity():
        return Xform()
    @staticmethod
    def translate(dx: float, dy: float, dz: float) -> 'Xform':
        M = np.eye(4); M[:3, 3] = [dx, dy, dz]; return Xform(M)
    @staticmethod
    def scale(s: float) -> 'Xform':
        M = np.diag([s, s, s, 1.0]); return Xform(M)
    @staticmethod
    def rotate_y(theta: float) -> 'Xform':
        c, s = math.cos(theta), math.sin(theta)
        M = np.array([[ c, 0.0,  s, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [-s, 0.0, c, 0.0],
                      [0.0,0.0,0.0,1.0]], dtype=np.float64)
        return Xform(M)
    def __matmul__(self, other: 'Xform') -> 'Xform':
        return Xform(self.M @ other.M)
    def inverse(self) -> 'Xform':
        return Xform(np.linalg.inv(self.M))

class Interval:
    __slots__ = ('lo','hi')
    def __init__(self, lo: float, hi: float):
        if hi < lo: lo, hi = hi, lo
        self.lo = float(lo); self.hi = float(hi)
    def width(self) -> float: return self.hi - self.lo
    def contains(self, x: float) -> bool: return self.lo <= x <= self.hi
    def intersect(self, other: 'Interval') -> 'Interval | None':
        lo = max(self.lo, other.lo); hi = min(self.hi, other.hi)
        return None if hi < lo else Interval(lo, hi)
    def expand(self, eps: float) -> 'Interval':
        return Interval(self.lo - eps, self.hi + eps)
    def __repr__(self): return f"Interval({self.lo},{self.hi})"
