from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Any, Dict


def _load_module_from_path(module_name: str, file_path: Path):
    loader = SourceFileLoader(module_name, str(file_path))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise RuntimeError(f"Failed to create module spec for {file_path}")
    module = module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Missing loader for {file_path}")
    # Ensure decorators (e.g. dataclasses) can resolve module globals during import.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_josephson(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]  # .../AdaptiveCAD
    examples_path = repo_root / "examples" / "adaptive_josephson_lattice.py"

    if not examples_path.exists():
        raise FileNotFoundError(str(examples_path))

    # Ensure the script can import its own local deps (if any).
    sys.path.insert(0, str(repo_root))

    mod = _load_module_from_path("adaptive_josephson_lattice", examples_path)

    cfg = getattr(mod, "SimulationConfig")(
        size=int(args.size),
        steps=int(args.steps),
        dt=float(args.dt),
        eta=float(args.eta),
        mu=float(args.mu),
        s_target=float(args.s_target),
        gamma=float(args.gamma),
        noise=float(args.noise),
        pi0=float(args.pi0),
        seed=None if args.seed is None else int(args.seed),
    )

    out = mod.run_simulation(cfg)

    import numpy as np  # local import so bridge can still be inspected without numpy

    result = {
        "config": asdict(cfg),
        "rho_s": float(out["rho_s"]),
        "Tc_proxy": float(out["Tc_proxy"]),
        "pi_mean": float(np.mean(out["pi_map"])),
        "S_mean": float(np.mean(out["S_map"])),
        "J_mean": float(np.mean(out["J_map"])),
    }

    return result


def run_pr_relaxation(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]  # .../AdaptiveCAD
    sys.path.insert(0, str(repo_root))

    from adaptivecad.pr import PRFieldConfig, relax_phase_field

    cfg = PRFieldConfig(
        size=int(args.size),
        steps=int(args.steps),
        dt=float(args.dt),
        diffusion=float(args.diffusion),
        coupling=float(args.coupling),
        coupling_mode=str(args.coupling_mode),
        phase_dim=int(args.phase_dim),
        phase_space=str(args.phase_space),
        seed=None if args.seed is None else int(args.seed),
    )

    metrics, _state = relax_phase_field(cfg)
    return metrics


def run_pr_and_export_stl(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]  # .../AdaptiveCAD
    sys.path.insert(0, str(repo_root))

    from adaptivecad.pr import PRFieldConfig, relax_phase_field, export_phase_field_as_heightmap_stl

    cfg = PRFieldConfig(
        size=int(args.size),
        steps=int(args.steps),
        dt=float(args.dt),
        diffusion=float(args.diffusion),
        coupling=float(args.coupling),
        coupling_mode=str(args.coupling_mode),
        phase_dim=int(args.phase_dim),
        phase_space=str(args.phase_space),
        seed=None if args.seed is None else int(args.seed),
    )

    metrics, state = relax_phase_field(cfg)

    # Export heightmap STL
    stl_bytes = export_phase_field_as_heightmap_stl(
        state.phi,
        scale_xy=float(args.scale_xy),
        scale_z=float(args.scale_z),
    )

    # Encode as base64 for JSON
    stl_b64 = base64.b64encode(stl_bytes).decode("ascii")

    return {
        **metrics,
        "stl_data": stl_b64,
        "stl_size_bytes": len(stl_bytes),
        "filename": f"pr_phase_field_{args.size}x{args.size}_{args.steps}steps.stl",
    }


def run_pr_and_export_ama(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]  # .../AdaptiveCAD
    sys.path.insert(0, str(repo_root))

    from adaptivecad.pr import PRFieldConfig, relax_phase_field, export_phase_field_as_ama

    cfg = PRFieldConfig(
        size=int(args.size),
        steps=int(args.steps),
        dt=float(args.dt),
        diffusion=float(args.diffusion),
        coupling=float(args.coupling),
        coupling_mode=str(args.coupling_mode),
        phase_dim=int(args.phase_dim),
        phase_space=str(args.phase_space),
        seed=None if args.seed is None else int(args.seed),
    )

    metrics, state = relax_phase_field(cfg)

    def _best_effort_git_commit(repo_root: Path) -> str:
        try:
            import subprocess

            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or "unknown"
        except Exception:
            return "unknown"

    def _best_effort_version() -> str:
        try:
            import adaptivecad  # type: ignore

            v = getattr(adaptivecad, "__version__", None)
            return str(v) if v is not None else "unknown"
        except Exception:
            return "unknown"

    cosmo_meta = {
        "ama_version": "0.1",
        "model": {
            "name": "AdaptiveCAD-PR",
            "pr_root": {
                "pi_a": "adaptive",
                "theta_cut": "(-pi,pi]" if str(cfg.phase_space) == "wrapped" else "unwrapped",
                "epsilon_tie": "1e-9",
                "delta_theta_max": "pi",
                "invariants": ["2pi_a", "w", "b", "C"],
            },
        },
        "run": {
            "solver": "adaptivecad.pr.relax_phase_field",
            "solver_version": _best_effort_version(),
            "git_commit": _best_effort_git_commit(repo_root),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": None if args.seed is None else int(args.seed),
            "platform": {
                "python": sys.version.split()[0],
                "platform": os.name,
            },
        },
        "numerics": {
            "grid": [int(cfg.size), int(cfg.size)],
            "integrator": "explicit_euler",
            "step_size": float(cfg.dt),
            "tolerances": {"abs": None, "rel": None},
            "boundary_conditions": "periodic",
            "steps": int(cfg.steps),
        },
        "observables": {
            "falsifier": {
                "map_units": "rad",
                "summary_stat": [
                    {"name": "mean", "value": float(metrics.get("falsifier_mean", 0.0)), "uncertainty": None},
                    {"name": "max", "value": float(metrics.get("falsifier_max", 0.0)), "uncertainty": None},
                ],
                "definition": "plaquette loop closure residual on phase differences",
            }
        },
        "tests": [
            {
                "id": "T0_falsifier_contract",
                "metric": "falsifier_max",
                "value": float(metrics.get("falsifier_max", 0.0)),
                "threshold": None,
                "status": "UNKNOWN",
                "definition": "max over grid of plaquette loop closure residual",
                "units": "rad",
                "acceptance_region": None,
            }
        ],
    }

    filename = f"pr_phase_field_{args.size}x{args.size}_{args.steps}steps.ama"
    ama_bytes = export_phase_field_as_ama(
        state.phi,
        falsifier_residual=state.falsifier_residual,
        cosmo_meta=cosmo_meta,
        scale_xy=float(args.scale_xy),
        scale_z=float(args.scale_z),
        filename=filename,
        params={
            "dt": float(args.dt),
            "diffusion": float(args.diffusion),
            "coupling": float(args.coupling),
            "coupling_mode": str(args.coupling_mode),
            "phase_dim": int(args.phase_dim),
            "phase_space": str(args.phase_space),
            "seed": None if args.seed is None else int(args.seed),
        },
        units=str(args.units),
        defl=float(args.defl),
    )

    # Write to disk in the current working directory for convenience.
    out_path = Path.cwd() / filename
    out_path.write_bytes(ama_bytes)

    return_mode = getattr(args, "return_mode", "inline")
    if return_mode == "path":
        return {
            **metrics,
            "ok": True,
            "ama_size_bytes": int(len(ama_bytes)),
            "filename": filename,
            "saved_path": str(out_path),
        }

    ama_b64 = base64.b64encode(ama_bytes).decode("ascii")
    return {
        **metrics,
        "ok": True,
        "ama_data": ama_b64,
        "ama_size_bytes": int(len(ama_bytes)),
        "filename": filename,
        "saved_path": str(out_path),
    }


def gen_blackholes_ama(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]  # .../AdaptiveCAD
    sys.path.insert(0, str(repo_root))

    # Import generator logic
    gen_path = repo_root / "apps-sdk" / "generate_binary_blackholes_ama.py"
    mod = _load_module_from_path("generate_binary_blackholes_ama", gen_path)

    size = int(args.size)
    scale_xy = float(args.scale_xy)
    scale_z = float(args.scale_z)

    field = mod.make_binary_blackholes_field(size)

    import tempfile
    import base64

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ama")
    tmp_path = Path(tmp.name)
    tmp.close()
    mod.write_ama(tmp_path, field, scale_xy=scale_xy, scale_z=scale_z)
    ama_bytes = tmp_path.read_bytes()
    ama_b64 = base64.b64encode(ama_bytes).decode("ascii")

    return {
        "ok": True,
        "filename": str(tmp_path.name),
        "ama_data": ama_b64,
        "ama_size_bytes": int(len(ama_bytes)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="adaptivecad_bridge", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    josephson = sub.add_parser("run_josephson", help="Run Josephson lattice headlessly")
    josephson.add_argument("--size", type=int, default=48)
    josephson.add_argument("--steps", type=int, default=400)
    josephson.add_argument("--dt", type=float, default=0.05)
    josephson.add_argument("--eta", type=float, default=0.6)
    josephson.add_argument("--mu", type=float, default=0.10)
    josephson.add_argument("--s-target", dest="s_target", type=float, default=0.50)
    josephson.add_argument("--gamma", type=float, default=0.5)
    josephson.add_argument("--noise", type=float, default=0.03)
    josephson.add_argument("--pi0", type=float, default=1.0)
    josephson.add_argument("--seed", type=int, default=0)

    pr = sub.add_parser("run_pr_relaxation", help="Run Phase-Resolved (PR) field relaxation")
    pr.add_argument("--size", type=int, default=64)
    pr.add_argument("--steps", type=int, default=200)
    pr.add_argument("--dt", type=float, default=0.15)
    pr.add_argument("--diffusion", type=float, default=0.35)
    pr.add_argument("--coupling", type=float, default=0.25)
    pr.add_argument("--coupling-mode", dest="coupling_mode", choices=("none", "geom_target"), default="geom_target")
    pr.add_argument("--phase-dim", dest="phase_dim", type=int, default=2)
    pr.add_argument("--phase-space", dest="phase_space", choices=("unwrapped", "wrapped"), default="unwrapped")
    pr.add_argument("--seed", type=int, default=0)

    pr_stl = sub.add_parser("run_pr_and_export_stl", help="Run PR relaxation and export as STL heightmap")
    pr_stl.add_argument("--size", type=int, default=64)
    pr_stl.add_argument("--steps", type=int, default=200)
    pr_stl.add_argument("--dt", type=float, default=0.15)
    pr_stl.add_argument("--diffusion", type=float, default=0.35)
    pr_stl.add_argument("--coupling", type=float, default=0.25)
    pr_stl.add_argument("--coupling-mode", dest="coupling_mode", choices=("none", "geom_target"), default="geom_target")
    pr_stl.add_argument("--phase-dim", dest="phase_dim", type=int, default=2)
    pr_stl.add_argument("--phase-space", dest="phase_space", choices=("unwrapped", "wrapped"), default="unwrapped")
    pr_stl.add_argument("--scale-xy", dest="scale_xy", type=float, default=1.0)
    pr_stl.add_argument("--scale-z", dest="scale_z", type=float, default=1.0)
    pr_stl.add_argument("--seed", type=int, default=0)

    pr_ama = sub.add_parser("run_pr_and_export_ama", help="Run PR relaxation and export as AdaptiveCAD AMA (uses OCC kernel)")
    pr_ama.add_argument("--size", type=int, default=64)
    pr_ama.add_argument("--steps", type=int, default=200)
    pr_ama.add_argument("--dt", type=float, default=0.15)
    pr_ama.add_argument("--diffusion", type=float, default=0.35)
    pr_ama.add_argument("--coupling", type=float, default=0.25)
    pr_ama.add_argument("--coupling-mode", dest="coupling_mode", choices=("none", "geom_target"), default="geom_target")
    pr_ama.add_argument("--phase-dim", dest="phase_dim", type=int, default=2)
    pr_ama.add_argument("--phase-space", dest="phase_space", choices=("unwrapped", "wrapped"), default="unwrapped")
    pr_ama.add_argument("--scale-xy", dest="scale_xy", type=float, default=1.0)
    pr_ama.add_argument("--scale-z", dest="scale_z", type=float, default=1.0)
    pr_ama.add_argument("--units", type=str, default="mm")
    pr_ama.add_argument("--defl", type=float, default=0.05)
    pr_ama.add_argument("--seed", type=int, default=0)
    pr_ama.add_argument("--return", dest="return_mode", choices=("inline", "path"), default="inline")

    bh = sub.add_parser("gen_blackholes_ama", help="Generate a binary blackholes analytic AMA")
    bh.add_argument("--size", type=int, default=128)
    bh.add_argument("--scale-xy", dest="scale_xy", type=float, default=1.0)
    bh.add_argument("--scale-z", dest="scale_z", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    # Force headless plotting if any imported modules touch matplotlib.
    os.environ.setdefault("MPLBACKEND", "Agg")

    args = parse_args()

    if args.cmd == "run_josephson":
        result = run_josephson(args)
    elif args.cmd == "run_pr_relaxation":
        result = run_pr_relaxation(args)
    elif args.cmd == "run_pr_and_export_stl":
        result = run_pr_and_export_stl(args)
    elif args.cmd == "run_pr_and_export_ama":
        result = run_pr_and_export_ama(args)
    elif args.cmd == "gen_blackholes_ama":
        result = gen_blackholes_ama(args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
