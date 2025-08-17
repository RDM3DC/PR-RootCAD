import argparse, numpy as np
from .quant import quantize, dequantize
from .zeros import zerosuppress, unsuppress

def quantize_main(argv=None):
    ap = argparse.ArgumentParser(description="GPUC quantize (CPU-safe)")
    ap.add_argument("--in", dest="infile", required=True, help="Input .npy")
    ap.add_argument("--out", required=True, help="Output .npz")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--block", type=int, default=0, help="Block count (0 for global scale)")
    args = ap.parse_args(argv)
    x = np.load(args.infile).astype(np.float32)
    pkt = quantize(x, bits=args.bits, block=args.block)
    # save arrays cleanly
    if "q" in pkt and isinstance(pkt["q"], np.ndarray):
        np.savez_compressed(args.out, **{k: v for k, v in pkt.items() if isinstance(v, (np.ndarray, int, float, tuple, list))})
    else:
        np.savez_compressed(args.out, **pkt)

def dequantize_main(argv=None):
    ap = argparse.ArgumentParser(description="GPUC dequantize")
    ap.add_argument("--in", dest="infile", required=True, help="Input .npz produced by quantize")
    ap.add_argument("--out", required=True, help="Output .npy")
    args = ap.parse_args(argv)
    data = np.load(args.infile, allow_pickle=True)
    pkt = {k: data[k] for k in data.files}
    # ensure python types
    if "shape" in pkt: pkt["shape"] = tuple(pkt["shape"])
    y = dequantize(pkt)
    np.save(args.out, y)

def zerosuppress_main(argv=None):
    ap = argparse.ArgumentParser(description="GPUC zero-suppress")
    ap.add_argument("--in", dest="infile", required=True, help="Input .npy")
    ap.add_argument("--out", required=True, help="Output .npz")
    ap.add_argument("--eps", type=float, default=0.0)
    args = ap.parse_args(argv)
    x = np.load(args.infile)
    pkt = zerosuppress(x, eps=args.eps)
    np.savez_compressed(args.out, **pkt)

def unsuppress_main(argv=None):
    ap = argparse.ArgumentParser(description="GPUC unsuppress")
    ap.add_argument("--in", dest="infile", required=True, help="Input .npz produced by zerosuppress")
    ap.add_argument("--out", required=True, help="Output .npy")
    args = ap.parse_args(argv)
    data = np.load(args.infile, allow_pickle=True)
    pkt = {k: data[k] for k in data.files}
    pkt["shape"] = tuple(pkt["shape"])
    pkt["idx"] = pkt["idx"]
    pkt["vals"] = pkt["vals"]
    y = unsuppress(pkt)
    np.save(args.out, y)
