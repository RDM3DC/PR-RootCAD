import numpy as np
from cmc.one_d import encode_1d, decode_1d

def test_cmc_1d_psnr():
    x = np.sin(np.linspace(0, 8*np.pi, 2000)).astype(np.float32)
    pkg = encode_1d(x, tau=0.01, max_err=0.01)
    y = decode_1d(pkg, n=len(x), alpha=0.2, mu=0.01)
    mse = np.mean((x - y)**2)
    psnr = 10*np.log10(1.0 / (mse + 1e-12))
    assert psnr > 20  # decent reconstruction

def test_cmc_anchor_coverage():
    x = np.linspace(0, 1, 1000).astype(np.float32)
    pkg = encode_1d(x, tau=0.0, max_err=0.0001)
    # should be small number of anchors on straight line
    assert len(pkg["anchors"]) <= 5
