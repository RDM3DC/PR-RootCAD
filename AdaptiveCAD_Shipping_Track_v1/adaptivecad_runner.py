#!/usr/bin/env python3
import argparse, subprocess, sys, json, shlex
from pathlib import Path

def run(cmd):
    print(">>", cmd)
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode, proc.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="hotspot", choices=["hotspot","shear_band","rim","multi_hotspot","uniform"])
    ap.add_argument("--outdir", default="runs/out")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--density_max", type=float, default=0.15)
    ap.add_argument("--tau_max", type=float, default=4.0)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Optimizer (coarse → refine) with plots + GIF
    opt_cmd = f"python adaptivecad_fields_optimizer.py --scenario {args.scenario} --seeds " + " ".join(map(str,args.seeds)) + \
              f" --density_max {args.density_max} --tau_max {args.tau_max} --outdir {out}"
    if args.publish:
        opt_cmd += " --publish"
    rc, out_txt = run(opt_cmd)
    if rc != 0:
        sys.exit(rc)

    # 2) Redshift infuser — prints JSON and writes CSV
    inf_cmd = f"python adaptivecad_redshift_infuser.py --scenario {args.scenario} --outdir {out}"
    if args.publish:
        inf_cmd += " --publish"
    rc, inf_txt = run(inf_cmd)
    if rc != 0:
        sys.exit(rc)

    print("\n=== DONE ===")
    print(f"Artifacts in: {out}")
    print("Search for: entangled_sweep_metrics.csv, summary_* .csv, figs/*.png, figs/*.gif, redshift_selection.csv")

if __name__ == '__main__':
    main()