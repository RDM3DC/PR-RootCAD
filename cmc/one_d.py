import numpy as np
from typing import Dict, Any

def _select_anchors_1d(x: np.ndarray, tau: float = 0.01, max_err: float = 0.01):
    n = len(x)
    anchors = [(0, float(x[0]))]
    # Greedy pass: ensure linear interpolation error below max_err; add points where curvature > tau
    i = 1
    last_idx = 0
    while i < n - 1:
        # curvature proxy: second difference
        curv = abs(x[i-1] - 2*x[i] + x[i+1])
        if curv >= tau:
            anchors.append((i, float(x[i])))
            last_idx = i
        else:
            # check linear interpolation error from last anchor to i+1
            j = i + 1
            xi, yi = last_idx, x[last_idx]
            xj, yj = j, x[j]
            # predict at i
            if j != xi:
                y_lin = yi + (yj - yi) * (i - xi) / (j - xi)
                err = abs(y_lin - x[i])
                if err > max_err:
                    anchors.append((i, float(x[i])))
                    last_idx = i
        i += 1
    if anchors[-1][0] != n-1:
        anchors.append((n-1, float(x[-1])))
    return anchors

def encode_1d(x: np.ndarray, tau: float = 0.01, max_err: float = 0.01) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float32)
    anchors = _select_anchors_1d(x, tau=tau, max_err=max_err)
    return {"type": "1d", "anchors": anchors}

def _arp_smoother_step(y, target, alpha=0.2, mu=0.01):
    # discrete ARP-style: y_{t+1} = y_t + alpha*sign(target - y_t) - mu*y_t
    e = target - y
    return y + alpha*np.sign(e) - mu*y

def decode_1d(pkg: Dict[str, Any], n: int, alpha: float = 0.2, mu: float = 0.01) -> np.ndarray:
    assert pkg["type"] == "1d"
    anchors = pkg["anchors"]
    anchors = sorted(anchors, key=lambda p: p[0])
    y = np.zeros(n, dtype=np.float32)

    for k in range(len(anchors)-1):
        i0, v0 = anchors[k]
        i1, v1 = anchors[k+1]
        length = max(1, i1 - i0)
        for t, idx in enumerate(range(i0, i1+1)):
            # linear target
            target = v0 + (v1 - v0) * (t / length)
            if idx == i0:
                y[idx] = v0
            else:
                y[idx] = _arp_smoother_step(y[idx-1], target, alpha=alpha, mu=mu)
    return y
