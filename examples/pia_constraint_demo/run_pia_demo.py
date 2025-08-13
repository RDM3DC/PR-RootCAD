#!/usr/bin/env python3
"""
πₐ constraint-tunnel demo:
- Torus πₐ field viz (PNG)
- Planar πₐ "circle" toolpath vs Euclid (PNG)
- G-code (G1 polyline) for πₐ path
"""
import numpy as np, math, argparse
import matplotlib.pyplot as plt
from pathlib import Path

def trC_xy(x,y):
    g1 = np.exp(-((x-15)**2 + (y-5)**2)/(2*8**2))
    g2 = -0.8*np.exp(-((x+10)**2 + (y+12)**2)/(2*10**2))
    return 0.6*g1 + 0.6*g2

def generate_pia_path(R0=30.0, N=720, lam_field=0.04):
    pi = math.pi
    ds = 2*pi*R0 / N
    theta = 0.0
    xs, ys = [], []
    for k in range(N):
        x = R0*math.cos(theta); y = R0*math.sin(theta)
        xs.append(x); ys.append(y)
        pia_loc = pi * (1 + lam_field*trC_xy(x,y))
        dtheta = (pi/pia_loc) * (ds/R0)
        theta += dtheta
    xs.append(xs[0]); ys.append(ys[0])
    return np.array(xs), np.array(ys)

def save_gcode(xs, ys, out_path, feed=1200):
    lines = ["; π_a demo G-code — adaptive curvature", "G90", "G21", "G0 X0 Y0", f"G1 F{feed}"]
    for x,y in zip(xs, ys):
        lines.append(f"G1 X{round(float(x),3)} Y{round(float(y),3)}")
    Path(out_path).write_text("\n".join(lines)+"\n", encoding="utf-8")

def plot_toolpath(xs, ys, R0, out_png):
    import numpy as np, matplotlib.pyplot as plt, math
    t = np.linspace(0, 2*math.pi, 400)
    plt.figure(figsize=(6,6))
    plt.plot(R0*np.cos(t), R0*np.sin(t), label="Euclidean circle")
    plt.plot(xs, ys, label="πₐ-deformed path")
    plt.axis('equal'); plt.xlabel("X (mm)"); plt.ylabel("Y (mm)"); plt.title("Planar toolpath: πₐ curvature adaptation")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--lam", type=float, default=0.04)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    xs, ys = generate_pia_path(args.radius, args.steps, args.lam)
    save_gcode(xs, ys, out/"pia_demo_path.gcode")
    plot_toolpath(xs, ys, args.radius, out/"pia_toolpath_preview.png")
    print("[ok] wrote", out)

if __name__ == "__main__":
    main()
