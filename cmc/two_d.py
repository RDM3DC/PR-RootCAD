import numpy as np
from typing import Dict, Any, List, Tuple

def _turning_angle(p_prev, p, p_next):
    v1 = p - p_prev
    v2 = p_next - p
    n1 = np.linalg.norm(v1) + 1e-12
    n2 = np.linalg.norm(v2) + 1e-12
    cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
    return np.arccos(cosang)

def _select_anchors_2d(points: np.ndarray, tau_rad: float = 0.05, max_err: float = 0.01):
    m = len(points)
    anchors = [(0, points[0].tolist())]
    last_idx = 0
    for i in range(1, m-1):
        ang = _turning_angle(points[i-1], points[i], points[i+1])
        if ang >= tau_rad:
            anchors.append((i, points[i].tolist()))
            last_idx = i
        else:
            # deviation from straight line segment
            j = i+1
            p0 = points[last_idx]
            p1 = points[j]
            if np.linalg.norm(p1 - p0) > 1e-12:
                t = (i - last_idx) / (j - last_idx)
                pred = p0 + t*(p1 - p0)
                err = np.linalg.norm(points[i] - pred)
                if err > max_err:
                    anchors.append((i, points[i].tolist()))
                    last_idx = i
    if anchors[-1][0] != m-1:
        anchors.append((m-1, points[-1].tolist()))
    return anchors

def encode_2d(points: np.ndarray, tau_rad: float = 0.05, max_err: float = 0.01) -> Dict[str, Any]:
    points = np.asarray(points, dtype=np.float32)
    anchors = _select_anchors_2d(points, tau_rad=tau_rad, max_err=max_err)
    return {"type": "2d", "anchors": anchors}

def _arp_step_2d(prev, target, alpha=0.2, mu=0.01):
    e = target - prev
    step = alpha * np.sign(e) - mu * prev
    return prev + step

def decode_2d(pkg: Dict[str, Any], m: int, alpha: float = 0.2, mu: float = 0.01) -> np.ndarray:
    assert pkg["type"] == "2d"
    anchors = sorted(pkg["anchors"], key=lambda p: p[0])
    out = np.zeros((m, 2), dtype=np.float32)
    for k in range(len(anchors)-1):
        i0, p0 = anchors[k]; p0 = np.array(p0, dtype=np.float32)
        i1, p1 = anchors[k+1]; p1 = np.array(p1, dtype=np.float32)
        length = max(1, i1 - i0)
        for t, idx in enumerate(range(i0, i1+1)):
            target = p0 + (p1 - p0) * (t / length)
            if idx == i0:
                out[idx] = p0
            else:
                out[idx] = _arp_step_2d(out[idx-1], target, alpha=alpha, mu=mu)
    return out
