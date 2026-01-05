from typing import Any, Dict, Tuple

import numpy as np


def _quant_params(arr: np.ndarray, bits: int = 8) -> Tuple[float, float]:
    # symmetric quantization: scale only
    maxv = float(np.max(np.abs(arr))) + 1e-12
    qmax = 2 ** (bits - 1) - 1
    scale = maxv / qmax
    return scale, 0.0


def quantize(arr: np.ndarray, bits: int = 8, block: int = 0) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float32)
    if block and block > 1:
        # per-block scales
        h, w = arr.shape[-2], arr.shape[-1]
        bh = max(1, h // block)
        bw = max(1, w // block)
        q = np.empty_like(arr, dtype=np.int8)
        scales = []
        for i in range(0, h, bh):
            for j in range(0, w, bw):
                tile = arr[..., i : i + bh, j : j + bw]
                s, _ = _quant_params(tile, bits=bits)
                scales.append(s)
                q[..., i : i + bh, j : j + bw] = np.clip(np.round(tile / s), -127, 127).astype(
                    np.int8
                )
        return {
            "mode": "quant",
            "bits": bits,
            "shape": arr.shape,
            "q": q,
            "scales": np.array(scales, dtype=np.float32),
            "block": (bh, bw),
        }
    else:
        s, _ = _quant_params(arr, bits=8)
        q = np.clip(np.round(arr / s), -127, 127).astype(np.int8)
        return {"mode": "quant", "bits": bits, "shape": arr.shape, "q": q, "scale": s}


def dequantize(pkt: Dict[str, Any]) -> np.ndarray:
    if pkt["mode"] != "quant":
        raise ValueError("Not a quantized packet")
    if "scale" in pkt:
        s = float(pkt["scale"])
        return (pkt["q"].astype(np.float32)) * s
    else:
        q = pkt["q"].astype(np.float32)
        scales = pkt["scales"].astype(np.float32)
        bh, bw = pkt["block"]
        h, w = pkt["shape"][-2], pkt["shape"][-1]
        out = np.empty(pkt["shape"], dtype=np.float32)
        idx = 0
        for i in range(0, h, bh):
            for j in range(0, w, bw):
                s = float(scales[idx])
                idx += 1
                out[..., i : i + bh, j : j + bw] = q[..., i : i + bh, j : j + bw] * s
        return out
