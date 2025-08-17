
from typing import List, Tuple
import math

Point = Tuple[float, float]

def _perp_dist(p: Point, a: Point, b: Point) -> float:
    (x, y), (x1, y1), (x2, y2) = p, a, b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1)*dx + (y - y1)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    px = x1 + t*dx
    py = y1 + t*dy
    return math.hypot(x - px, y - py)

def _seg_len(a: Point, b: Point) -> float:
    return math.hypot(b[0]-a[0], b[1]-a[1])

def adaptive_cad(anchors: List[List[float]], max_err: float=3.0, min_seg: float=4.0, max_seg: float=40.0) -> List[List[float]]:
    pts = [(float(x), float(y)) for x, y in anchors]
    if len(pts) <= 2: return [[x,y] for x,y in pts]

    def recurse(i0: int, i1: int, out: List[Point]):
        a = pts[i0]; b = pts[i1]
        # find farthest point index between i0 and i1
        imax = -1; dmax = -1.0
        for i in range(i0+1, i1):
            d = _perp_dist(pts[i], a, b)
            if d > dmax:
                dmax = d; imax = i
        seglen = _seg_len(a, b)
        need_split = (dmax > max_err) or (seglen > max_seg)
        if need_split and imax != -1:
            recurse(i0, imax, out)
            recurse(imax, i1, out)
        else:
            # accept segment, but if very short, we still keep end to maintain topology
            out.append(b)

    out = [pts[0]]
    recurse(0, len(pts)-1, out)
    # optional pass to merge segments shorter than min_seg by removing intermediate points if possible
    if len(out) > 2 and min_seg > 0:
        merged = [out[0]]
        for i in range(1, len(out)-1):
            prev = merged[-1]; cur = out[i]; nxt = out[i+1]
            if _seg_len(prev, cur) < min_seg and _seg_len(cur, nxt) < min_seg:
                # try skipping cur
                merged.append(nxt)
            else:
                merged.append(cur)
        if merged[-1] != out[-1]:
            merged.append(out[-1])
        out = merged
    return [[x,y] for x,y in out]
