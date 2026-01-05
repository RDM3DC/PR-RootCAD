from typing import Any, Dict

import numpy as np


def _select_anchors_1d(x: np.ndarray, tau: float = 0.01, max_err: float = 0.01):
    n = len(x)
    if n <= 2:
        return [(0, float(x[0])), (n - 1, float(x[-1]))]
    anchors = [(0, float(x[0]))]
    i0 = 0
    j = i0 + 1
    # Greedy segment growth ensuring max deviation <= max_err
    while j < n:
        i1 = j
        v0 = float(x[i0])
        v1 = float(x[i1])
        length = max(1, i1 - i0)
        max_dev = 0.0
        # compute deviation within the current segment [i0, i1]
        for k in range(i0 + 1, i1):
            t = (k - i0) / length
            y_lin = v0 + (v1 - v0) * t
            dev = abs(float(x[k]) - y_lin)
            if dev > max_dev:
                max_dev = dev
                if max_dev > max_err:
                    break
        if max_dev <= max_err:
            j += 1
        else:
            # finalize previous point as anchor and restart from there
            anchors.append((i1 - 1, float(x[i1 - 1])))
            i0 = i1 - 1
            j = i0 + 1
    if anchors[-1][0] != n - 1:
        anchors.append((n - 1, float(x[-1])))

    # Reduce redundant anchors that are collinear within tolerance
    def collinear(a, b, c, tol=max_err * 0.5):
        (i0, v0), (i1, v1), (i2, v2) = a, b, c
        if i2 == i0:
            return True
        t = (i1 - i0) / (i2 - i0)
        v_lin = v0 + (v2 - v0) * t
        return abs(v_lin - v1) <= tol

    pruned = [anchors[0]]
    for k in range(1, len(anchors) - 1):
        if not collinear(pruned[-1], anchors[k], anchors[k + 1]):
            pruned.append(anchors[k])
    pruned.append(anchors[-1])
    return pruned


def encode_1d(x: np.ndarray, tau: float = 0.01, max_err: float = 0.01) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float32)
    anchors = _select_anchors_1d(x, tau=tau, max_err=max_err)
    return {"type": "1d", "anchors": anchors}


def _arp_smoother_step(y, target, alpha=0.2, mu=0.01):
    # discrete ARP-style: y_{t+1} = y_t + alpha*(target - y_t) - mu*y_t
    # use true proportional error to improve fidelity
    e = target - y
    return y + alpha * e - mu * y


def decode_1d(pkg: Dict[str, Any], n: int, alpha: float = 0.2, mu: float = 0.01) -> np.ndarray:
    assert pkg["type"] == "1d"
    anchors = pkg["anchors"]
    anchors = sorted(anchors, key=lambda p: p[0])
    y = np.zeros(n, dtype=np.float32)

    # Always decode via piecewise-linear interpolation between anchors for accuracy
    for k in range(len(anchors) - 1):
        i0, v0 = anchors[k]
        i1, v1 = anchors[k + 1]
        length = max(1, i1 - i0)
        for t, idx in enumerate(range(i0, i1 + 1)):
            y[idx] = v0 + (v1 - v0) * (t / length)
    return y
