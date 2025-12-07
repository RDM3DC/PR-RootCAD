import numpy as np

from gpuc.quant import dequantize, quantize
from gpuc.zeros import unsuppress, zerosuppress


def test_gpuc_quant_roundtrip():
    x = np.random.randn(256, 256).astype(np.float32) * 0.25
    pkt = quantize(x, bits=8)
    y = dequantize(pkt)
    # quantization error acceptable
    rel = np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-9)
    assert rel < 0.1


def test_gpuc_zero_suppress():
    x = np.zeros((64, 64), dtype=np.float32)
    x[10, 10] = 1.0
    x[20, 30] = -2.0
    pkt = zerosuppress(x, eps=0.0)
    y = unsuppress(pkt)
    assert np.allclose(x, y)
