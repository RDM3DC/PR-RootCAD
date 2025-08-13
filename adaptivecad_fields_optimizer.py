#!/usr/bin/env python3
"""
AdaptiveCAD Fields Optimizer (ROS2 or Offline)
------------------------------------------------
Sweeps correlation ``rho`` over correlated λ–α fields and applies scenario
modifiers.  Two stage search: a coarse sweep followed by a refinement around the
best candidate that satisfies density and tortuosity constraints.  Metrics are
exported as CSVs, plots and an optional GIF.  If ``--publish`` is supplied, the
chosen row is published on ``/adaptivecad/fields`` using ROS2 (falls back to
offline if ROS2 is missing).

Usage (hotspot example):

```
python adaptivecad_fields_optimizer.py \
  --scenario hotspot --seeds 0 1 2 \
  --coarse 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --refine_width 0.2 --refine_step 0.05 \
  --density_max 0.15 --tau_max 4.0 \
  --outdir results_hotspot
```
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:  # pragma: no cover - optional
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional
    imageio = None

# Reuse helpers from entangled fields prototype
from adaptivecad_entangled_fields import (
    correlated_fields,
    inside_hex_xy,
    wire_metrics,
)


# ---------------------------------------------------------------------------
# Scenario modifiers
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, y: np.ndarray, x0: float, y0: float, s: float) -> np.ndarray:
    return np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * s ** 2)))


def apply_scenario(lam: np.ndarray, alpha: np.ndarray, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    """Apply simple analytic scenarios to the λ–α fields.

    Scenarios:
      - hotspot: single Gaussian bump in the centre
      - shear_band: linear gradient along x
      - rim: ring-like increase in the middle radius
      - multi_hotspot: two Gaussian bumps
      - uniform: no modification
    """

    nx, ny = lam.shape
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    if scenario == "hotspot":
        g = _gaussian(X, Y, 0.0, 0.0, 0.25)
        lam += g
        alpha += g
    elif scenario == "shear_band":
        g = (X > 0).astype(float)
        lam += g * 0.5
    elif scenario == "rim":
        R = np.sqrt(X ** 2 + Y ** 2)
        g = np.exp(-((R - 0.7) ** 2) / (2 * 0.1 ** 2))
        lam += g
        alpha += g
    elif scenario == "multi_hotspot":
        g = _gaussian(X, Y, -0.4, -0.4, 0.15) + _gaussian(X, Y, 0.4, 0.4, 0.15)
        lam += g
        alpha += g
    elif scenario == "uniform":
        pass
    else:  # pragma: no cover - user error
        raise ValueError(f"Unknown scenario '{scenario}'")

    lam = np.clip(lam, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    return lam, alpha


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_fields(
    rho: float,
    seed: int,
    mask: np.ndarray,
    scenario: str,
    nx: int,
    ny: int,
    sigma: float,
    R: float,
    r_wire0: float,
    t0: float,
    gam_w: float,
    gam_a: float,
    L: float,
    wires_per_axis: int,
    n_cells: int,
    amp: float,
    N: int,
    kappa: float,
) -> Dict[str, float]:
    """Generate fields, apply scenario and compute metrics."""
    lam_field, alpha_field = correlated_fields(nx, ny, corr=rho, sigma=sigma, seed=seed)
    lam_field, alpha_field = apply_scenario(lam_field, alpha_field, scenario)

    lam_vals = lam_field[mask]
    alpha_vals = alpha_field[mask]
    lam_mean = float(lam_vals.mean())
    alpha_mean = float(alpha_vals.mean())

    r_wire = r_wire0 * (1.0 + gam_w * lam_mean)
    t = t0 * (1.0 + gam_a * alpha_mean)
    wm = wire_metrics(
        L=L,
        wires_per_axis=wires_per_axis,
        r_wire=r_wire,
        n_cells=n_cells,
        amp=amp,
        N=N,
    )
    deltaT = 1.0 / (wm["A_eff"] + kappa * t)

    return {
        "rho": rho,
        "seed": seed,
        "lam_mean": lam_mean,
        "alpha_mean": alpha_mean,
        "r_wire": r_wire,
        "t": t,
        "density": wm["density"],
        "A_eff": wm["A_eff"],
        "tau": wm["tau"],
        "deltaT": deltaT,
    }


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def run_stage(rhos: Iterable[float], stage: str, args, mask, X, Y, figs_dir: Path) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Path]]:
    metrics: List[Dict[str, float]] = []
    summary: List[Dict[str, float]] = []
    panels: List[Path] = []

    for rho in rhos:
        stage_rows = []
        for seed in args.seeds:
            row = evaluate_fields(
                rho=rho,
                seed=seed,
                mask=mask,
                scenario=args.scenario,
                nx=args.nx,
                ny=args.ny,
                sigma=args.sigma,
                R=args.R,
                r_wire0=args.r_wire0,
                t0=args.t0,
                gam_w=args.gam_w,
                gam_a=args.gam_a,
                L=args.L,
                wires_per_axis=args.wires_per_axis,
                n_cells=args.n_cells,
                amp=args.amp,
                N=args.N,
                kappa=args.kappa,
            )
            metrics.append({"stage": stage, **row})
            stage_rows.append(row)

            # plotting individual fields for GIF panels
            lam_field, alpha_field = correlated_fields(args.nx, args.ny, corr=rho, sigma=args.sigma, seed=seed)
            lam_field, alpha_field = apply_scenario(lam_field, alpha_field, args.scenario)
            lam_plot = np.where(mask, lam_field, np.nan)
            alpha_plot = np.where(mask, alpha_field, np.nan)
            panel_path = figs_dir / f"{stage}_rho{rho:.3f}_seed{seed}.png"
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            im0 = axs[0].imshow(lam_plot, origin="lower", extent=(-args.R, args.R, -args.R, args.R), cmap="viridis")
            axs[0].set_title("λ")
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            im1 = axs[1].imshow(alpha_plot, origin="lower", extent=(-args.R, args.R, -args.R, args.R), cmap="plasma")
            axs[1].set_title("α")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            fig.suptitle(f"ρ={rho:.3f} seed={seed}")
            plt.tight_layout()
            fig.savefig(panel_path)
            plt.close(fig)
            panels.append(panel_path)

        # average over seeds for summary
        rho_metrics = {k: np.mean([r[k] for r in stage_rows]) for k in stage_rows[0].keys() if k not in {"seed"}}
        rho_metrics["stage"] = stage
        summary.append(rho_metrics)

    return metrics, summary, panels


def choose_best(summary: List[Dict[str, float]], density_max: float, tau_max: float) -> Dict[str, float]:
    valid = [r for r in summary if r["density"] <= density_max and r["tau"] <= tau_max]
    if not valid:
        return {}
    return min(valid, key=lambda r: r["deltaT"])


def plot_metric(summary_all: List[Dict[str, float]], outdir: Path, scenario: str) -> None:
    summary_all = sorted(summary_all, key=lambda r: r["rho"])
    rhos = [r["rho"] for r in summary_all]
    for key, fname, ylabel in [
        ("deltaT", "DeltaT_vs_rho", "ΔT proxy"),
        ("A_eff", "Aeff_vs_rho", "A_eff"),
        ("density", "density_vs_rho", "density"),
        ("tau", "tau_vs_rho", "τ"),
    ]:
        vals = [r[key] for r in summary_all]
        plt.figure(figsize=(4, 3))
        plt.plot(rhos, vals, "o-")
        plt.xlabel("ρ")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(outdir / f"{fname}_{scenario}.png")
        plt.close()


def build_gif(panels: List[Path], figs_dir: Path) -> None:
    if imageio is None:
        return
    try:  # pragma: no cover - optional
        images = [imageio.imread(p) for p in panels]
        imageio.mimsave(figs_dir / "rho_sweep_panels.gif", images, duration=0.8)
    except Exception:
        pass


def publish_json(row: Dict[str, float]) -> None:
    try:  # pragma: no cover - optional
        import rclpy
        from std_msgs.msg import String

        rclpy.init()
        node = rclpy.create_node("adaptivecad_fields_optimizer")
        pub = node.create_publisher(String, "/adaptivecad/fields", 10)
        msg = String()
        msg.data = json.dumps(row)
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()
    except Exception as exc:  # pragma: no cover - optional
        print(f"ROS2 publish failed: {exc}")


def main() -> None:  # pragma: no cover - CLI tool
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["hotspot", "shear_band", "rim", "multi_hotspot", "uniform"], default="hotspot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--coarse", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    ap.add_argument("--refine_width", type=float, default=0.2)
    ap.add_argument("--refine_step", type=float, default=0.05)
    ap.add_argument("--density_max", type=float, default=0.15)
    ap.add_argument("--tau_max", type=float, default=4.0)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--publish", action="store_true")
    # field generation parameters
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--R", type=float, default=25.0)
    ap.add_argument("--t0", type=float, default=2.0)
    ap.add_argument("--alpha0", type=float, default=0.6)
    ap.add_argument("--r_wire0", type=float, default=0.6)
    ap.add_argument("--gam_w", type=float, default=0.3)
    ap.add_argument("--gam_a", type=float, default=0.6)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--L", type=float, default=40.0)
    ap.add_argument("--wires_per_axis", type=int, default=3)
    ap.add_argument("--n_cells", type=int, default=3)
    ap.add_argument("--amp", type=float, default=4.0)
    ap.add_argument("--N", type=int, default=120)
    args = ap.parse_args()

    outdir: Path = args.outdir
    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # precompute mask and coordinate grids
    x = np.linspace(-args.R, args.R, args.nx)
    y = np.linspace(-args.R, args.R, args.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = inside_hex_xy(X, Y, args.R)

    # stage 1: coarse
    metrics_coarse, summary_coarse, panels = run_stage(args.coarse, "coarse", args, mask, X, Y, figs_dir)
    best_coarse = choose_best(summary_coarse, args.density_max, args.tau_max)
    if best_coarse:
        centre = best_coarse["rho"]
    else:
        centre = args.coarse[len(args.coarse) // 2]

    # stage 2: refine around coarse best
    r_start = max(0.0, centre - args.refine_width / 2)
    r_end = min(1.0, centre + args.refine_width / 2)
    refine_rhos = np.arange(r_start, r_end + 1e-9, args.refine_step)
    metrics_refine, summary_refine, panels_r = run_stage(refine_rhos, "refine", args, mask, X, Y, figs_dir)
    panels.extend(panels_r)

    # write CSVs
    all_metrics = metrics_coarse + metrics_refine
    with (outdir / "entangled_sweep_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    with (outdir / "summary_coarse.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_coarse[0].keys()))
        writer.writeheader()
        writer.writerows(summary_coarse)

    with (outdir / "summary_refine.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_refine[0].keys()))
        writer.writeheader()
        writer.writerows(summary_refine)

    # plots and gif
    summary_all = summary_coarse + summary_refine
    plot_metric(summary_all, outdir, args.scenario)
    build_gif(panels, figs_dir)

    # choose final best
    best_final = choose_best(summary_refine, args.density_max, args.tau_max)
    if not best_final:
        best_final = choose_best(summary_coarse, args.density_max, args.tau_max)
    if best_final:
        result = {
            "scenario": args.scenario,
            "rho_star": best_final["rho"],
            "density": best_final["density"],
            "tau": best_final["tau"],
            "deltaT": best_final["deltaT"],
        }
        print(json.dumps(result))
        if args.publish:
            publish_json(result)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
