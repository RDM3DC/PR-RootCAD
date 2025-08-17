import argparse, json, sys, numpy as np
from .one_d import encode_1d, decode_1d
from .two_d import encode_2d, decode_2d

def encode_1d_main(argv=None):
    ap = argparse.ArgumentParser(description="CMC encode 1D")
    ap.add_argument("--in", dest="infile", required=True, help="Input .npy (float array)")
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--max_err", type=float, default=0.01)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args(argv)
    x = np.load(args.infile).astype(np.float32)
    pkg = encode_1d(x, tau=args.tau, max_err=args.max_err)
    json.dump(pkg, open(args.out, "w"))

def decode_1d_main(argv=None):
    ap = argparse.ArgumentParser(description="CMC decode 1D")
    ap.add_argument("--in", dest="infile", required=True, help="Input JSON package")
    ap.add_argument("--n", type=int, required=True, help="Number of samples to reconstruct")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--out", required=True, help="Output .npy")
    args = ap.parse_args(argv)
    pkg = json.load(open(args.infile, "r"))
    y = decode_1d(pkg, n=args.n, alpha=args.alpha, mu=args.mu)
    np.save(args.out, y)

def encode_2d_main(argv=None):
    ap = argparse.ArgumentParser(description="CMC encode 2D paths")
    ap.add_argument("--in", dest="infile", required=True, help="Input .npy (N,2) float array")
    ap.add_argument("--tau_rad", type=float, default=0.05)
    ap.add_argument("--max_err", type=float, default=0.01)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args(argv)
    pts = np.load(args.infile).astype(np.float32)
    pkg = encode_2d(pts, tau_rad=args.tau_rad, max_err=args.max_err)
    json.dump(pkg, open(args.out, "w"))

def decode_2d_main(argv=None):
    ap = argparse.ArgumentParser(description="CMC decode 2D paths")
    ap.add_argument("--in", dest="infile", required=True, help="Input JSON package")
    ap.add_argument("--m", type=int, required=True, help="Number of points to reconstruct")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--out", required=True, help="Output .npy")
    args = ap.parse_args(argv)
    pkg = json.load(open(args.infile, "r"))
    out = decode_2d(pkg, m=args.m, alpha=args.alpha, mu=args.mu)
    import numpy as np
    np.save(args.out, out)
