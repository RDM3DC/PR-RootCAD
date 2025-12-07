#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def inside_hex_xy(X, Y, R):
    import numpy as _np

    return (_np.abs(X) <= R) & (_np.abs(X) + _np.sqrt(3.0) * _np.abs(Y) <= 2 * R)


def gaussian_kernel_fft(nx, ny, sigma):
    import numpy as _np

    kx = _np.fft.fftfreq(nx)
    ky = _np.fft.fftfreq(ny)
    KX, KY = _np.meshgrid(kx, ky, indexing="ij")
    H = _np.exp(-2 * (_np.pi**2) * (sigma**2) * (KX**2 + KY**2))
    return H


def correlated_fields(nx, ny, corr=0.25, sigma=6.0, seed=0):
    import numpy as _np

    rng = _np.random.default_rng(seed)
    u = rng.standard_normal((nx, ny))
    w = rng.standard_normal((nx, ny))
    v = corr * u + _np.sqrt(max(0.0, 1.0 - corr**2)) * w
    H = gaussian_kernel_fft(nx, ny, sigma)

    def blur(a):
        Af = _np.fft.fftn(a)
        return _np.fft.ifftn(Af * H).real

    u_b = blur(u)
    v_b = blur(v)

    def norm01(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-12)

    return norm01(u_b), norm01(v_b)


def apply_scenario(lam, alf, scenario, R, xs, ys):
    import numpy as _np

    X, Y = _np.meshgrid(xs, ys, indexing="ij")
    M = inside_hex_xy(X, Y, R)
    lam2, alf2 = lam.copy(), alf.copy()
    if scenario == "hotspot":
        bump = _np.exp(-(((X - 0.25 * R) ** 2 + (Y + 0.1 * R) ** 2) / (2 * (0.22 * R) ** 2)))
        alf2 = _np.clip(alf2 + 0.25 * bump, 0, 1)
    elif scenario == "shear_band":
        band = 0.5 * (_np.tanh((Y + 0.2 * R) / (0.05 * R)) - _np.tanh((Y - 0.2 * R) / (0.05 * R)))
        lam2 = _np.clip(lam2 + 0.20 * band, 0, 1)
    elif scenario == "rim":
        rim = (X**2 + Y**2) / (R**2)
        alf2 = _np.clip(alf2 + 0.20 * rim, 0, 1)
    elif scenario == "multi_hotspot":
        b1 = _np.exp(-(((X - 0.30 * R) ** 2 + (Y + 0.15 * R) ** 2) / (2 * (0.18 * R) ** 2)))
        b2 = _np.exp(-(((X + 0.25 * R) ** 2 + (Y - 0.20 * R) ** 2) / (2 * (0.16 * R) ** 2)))
        alf2 = _np.clip(alf2 + 0.18 * (b1 + b2), 0, 1)
    lam2 *= M
    alf2 *= M
    return lam2, alf2, M


def wire_gyroid_paths(L=40.0, n_cells=3, wires_per_axis=3, amp=4.0, N=120):
    import numpy as np

    xmin = ymin = zmin = -L / 2
    xmax = ymax = zmax = L / 2
    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)
    zs = np.linspace(zmin, zmax, N)
    k = 2 * math.pi * n_cells / L
    paths = []
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            y0 = ymin + (i + 0.5) * (L / wires_per_axis)
            z0 = zmin + (j + 0.5) * (L / wires_per_axis)
            P = np.stack(
                [xs, y0 + np.sin(k * xs + 0.6 * j) * 4.0, z0 + np.cos(k * xs + 0.4 * i) * 4.0],
                axis=1,
            )
            paths.append(P)
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            x0 = xmin + (i + 0.5) * (L / wires_per_axis)
            z0 = zmin + (j + 0.5) * (L / wires_per_axis)
            P = np.stack(
                [x0 + np.sin(k * ys + 0.5 * i) * 4.0, ys, z0 + np.cos(k * ys + 0.7 * j) * 4.0],
                axis=1,
            )
            paths.append(P)
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            x0 = xmin + (i + 0.5) * (L / wires_per_axis)
            y0 = ymin + (j + 0.5) * (L / wires_per_axis)
            P = np.stack(
                [x0 + np.sin(k * zs + 0.4 * j) * 4.0, y0 + np.cos(k * zs + 0.6 * i) * 4.0, zs],
                axis=1,
            )
            paths.append(P)
    return paths


def wire_metrics(L=40.0, wires_per_axis=3, r_wire=0.6, n_cells=3, N=120):
    import numpy as np

    paths = wire_gyroid_paths(L, n_cells, wires_per_axis, 4.0, N)
    total_len = 0.0
    taus = []
    for P in paths:
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()
        total_len += seg
        taus.append(seg / L)
    tau_mean = float(np.mean(taus))
    vol = math.pi * (r_wire**2) * total_len
    bbox_vol = L**3
    Aeff = vol / L
    density = vol / bbox_vol
    return dict(total_len=total_len, tau=tau_mean, A_eff=Aeff, density=density)


def choose_rho_with_guards(lam_mean, alf_mean, params):
    def eval_rho(rho):
        r_wire = params["r_wire0"] * (1.0 + params["gam_w"] * lam_mean)
        t_skin = params["t0"] * (1.0 + params["gam_a"] * alf_mean)
        met = wire_metrics(L=40.0, wires_per_axis=3, r_wire=r_wire)
        A_eff = met["A_eff"]
        density = met["density"]
        tau = met["tau"]
        dT = 1.0 / (A_eff + params["kappa"] * t_skin)
        return dict(
            rho=rho, r_wire=r_wire, t_skin=t_skin, A_eff=A_eff, density=density, tau=tau, DeltaT=dT
        )

    candidates = [0.25, 0.20, 0.30]
    trials = [eval_rho(r) for r in candidates]
    feas = [
        t
        for t in trials
        if (t["density"] <= params["density_max"] and t["tau"] <= params["tau_max"])
    ]
    if not feas:
        feas = trials
    best = sorted(feas, key=lambda t: t["DeltaT"])[0]
    return best, trials


def try_publish(topic, payload_json):
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except Exception:
        print(payload_json)
        return False
    rclpy.init()

    class Pub(Node):
        def __init__(self):
            super().__init__("adaptivecad_redshift_pub")
            self.pub = self.create_publisher(String, topic, 10)
            msg = String()
            msg.data = payload_json
            self.pub.publish(msg)
            self.get_logger().info(f"Published to {topic}")

    node = Pub()
    import rclpy as _r

    _r.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    _r.shutdown()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario",
        type=str,
        default="hotspot",
        choices=["hotspot", "shear_band", "rim", "multi_hotspot", "uniform"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=float, default=25.0)
    ap.add_argument("--grid", type=int, default=180)
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--outdir", type=str, default="redshift_results")
    ap.add_argument("--t0", type=float, default=2.0)
    ap.add_argument("--r_wire0", type=float, default=0.6)
    ap.add_argument("--gam_w", type=float, default=0.30)
    ap.add_argument("--gam_a", type=float, default=0.60)
    ap.add_argument("--kappa", type=float, default=0.4)
    ap.add_argument("--density_max", type=float, default=0.15)
    ap.add_argument("--tau_max", type=float, default=4.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nx = ny = args.grid
    xs = np.linspace(-args.R, args.R, nx)
    ys = np.linspace(-args.R, args.R, ny)
    lam, alf = correlated_fields(nx, ny, corr=0.25, sigma=args.sigma, seed=args.seed)
    lam, alf, M = apply_scenario(lam, alf, args.scenario, args.R, xs, ys)
    lam_mean = float(lam[M].mean()) if M.any() else 0.5
    alf_mean = float(alf[M].mean()) if M.any() else 0.5

    params = dict(
        t0=args.t0,
        r_wire0=args.r_wire0,
        gam_w=args.gam_w,
        gam_a=args.gam_a,
        kappa=args.kappa,
        density_max=args.density_max,
        tau_max=args.tau_max,
    )
    best, trials = choose_rho_with_guards(lam_mean, alf_mean, params)

    A_eff_vals = [t["A_eff"] for t in trials]
    t_vals = [t["t_skin"] for t in trials]
    A_min, A_ptp = min(A_eff_vals), (max(A_eff_vals) - min(A_eff_vals) + 1e-12)
    t_min, t_ptp = min(t_vals), (max(t_vals) - min(t_vals) + 1e-12)
    A_eff_n = (A_eff_vals[0] - A_min) / A_ptp
    t_n = (t_vals[0] - t_min) / t_ptp

    payload = {
        "scenario": args.scenario,
        "seed": args.seed,
        "rho_star": best["rho"],
        "mapping": {"r_wire_mm": best["r_wire"], "t_skin_mm": best["t_skin"]},
        "metrics": {
            "A_eff": best["A_eff"],
            "density": best["density"],
            "tau": best["tau"],
            "DeltaT": best["DeltaT"],
        },
        "weights": {"A_eff_norm": float(A_eff_n), "t_norm": float(t_n), "beta": args.kappa},
        "trials": trials,
    }
    js = json.dumps(payload)
    if args.publish:
        try_publish("/adaptivecad/redshift_config", js)
    else:
        print(js)

    import csv

    csv_path = outdir / "redshift_selection.csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "seed",
                "rho_star",
                "r_wire_mm",
                "t_skin_mm",
                "A_eff",
                "density",
                "tau",
                "DeltaT",
            ],
        )
        wcsv.writeheader()
        wcsv.writerow(
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "rho_star": best["rho"],
                "r_wire_mm": best["r_wire"],
                "t_skin_mm": best["t_skin"],
                "A_eff": best["A_eff"],
                "density": best["density"],
                "tau": best["tau"],
                "DeltaT": best["DeltaT"],
            }
        )
    print(str(csv_path))


if __name__ == "__main__":
    main()
