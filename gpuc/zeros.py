import numpy as np
from typing import Dict, Any

def zerosuppress(arr: np.ndarray, eps: float = 0.0) -> Dict[str, Any]:
    arr = np.asarray(arr)
    mask = np.abs(arr) > eps
    idx = np.flatnonzero(mask.ravel())
    vals = arr.ravel()[idx].astype(arr.dtype)
    return {"mode": "zerosuppress", "shape": arr.shape, "idx": idx.astype(np.int64), "vals": vals, "eps": float(eps)}

def unsuppress(pkt: Dict[str, Any]) -> np.ndarray:
    assert pkt["mode"] == "zerosuppress"
    out = np.zeros(pkt["shape"], dtype=pkt["vals"].dtype)
    out.ravel()[pkt["idx"].astype(np.int64)] = pkt["vals"]
    return out
