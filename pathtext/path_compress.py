import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _perp_dist(p: Point, a: Point, b: Point) -> float:
    (x, y), (x1, y1), (x2, y2) = p, a, b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px = x1 + t * dx
    py = y1 + t * dy
    return math.hypot(x - px, y - py)


def rdp(points: List[Point], eps: float) -> List[Point]:
    if len(points) < 3:
        return points[:]
    a, b = points[0], points[-1]
    max_d = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        d = _perp_dist(points[i], a, b)
        if d > max_d:
            max_d = d
            idx = i
    if max_d > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [a, b]


def adaptive_topk(points: List[Point], k: int) -> List[Point]:
    n = len(points)
    if n <= 2 or k <= 2:
        return [points[0], points[-1]] if n >= 2 else points[:]
    if n <= k:
        return points[:]
    # score by distance to chord (neighbor chord)
    scores = []
    for i in range(1, n - 1):
        d = _perp_dist(points[i], points[i - 1], points[i + 1])
        scores.append((d, i))
    idxs = [0, n - 1] + [i for _, i in sorted(scores, key=lambda t: t[0], reverse=True)[: k - 2]]
    idxs = sorted(set(idxs))
    return [points[i] for i in idxs]


def max_error(orig: List[Point], simp: List[Point]) -> float:
    if len(simp) < 2:
        x0, y0 = simp[0] if simp else (0.0, 0.0)
        return max((abs(x - x0) + abs(y - y0)) for (x, y) in orig) if orig else 0.0
    me = 0.0
    for p in orig:
        mind = 1e18
        for j in range(len(simp) - 1):
            d = _perp_dist(p, simp[j], simp[j + 1])
            if d < mind:
                mind = d
        if mind > me:
            me = mind
    return me


def compress_anchors(anchors: List[List[float]], eps_px: float) -> Dict:
    pts = [(float(x), float(y)) for x, y in anchors]
    comp = rdp(pts, eps_px)
    err = max_error(pts, comp) if len(pts) >= 2 else 0.0
    return {
        "method": "rdp",
        "eps_px": float(eps_px),
        "orig_points": len(pts),
        "new_points": len(comp),
        "max_err_px": float(err),
        "anchors": [[x, y] for x, y in comp],
    }


def compress_anchors_topk(anchors: List[List[float]], k: int) -> Dict:
    pts = [(float(x), float(y)) for x, y in anchors]
    comp = adaptive_topk(pts, k)
    err = max_error(pts, comp) if len(pts) >= 2 else 0.0
    return {
        "method": "topk",
        "k": int(k),
        "orig_points": len(pts),
        "new_points": len(comp),
        "max_err_px": float(err),
        "anchors": [[x, y] for x, y in comp],
    }
