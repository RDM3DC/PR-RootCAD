
# AdaptiveCAD — Shipping Track Pack (v1)

This pack gives you a crisp, **productizable** subset:
- **Fields Optimizer** — sweeps/auto-optimizes entangled λ–α fields (ρ) per scenario.
- **Redshift Energy Infuser** — picks ρ* with guardrails and emits a JSON config.
- **One‑shot Runner** — orchestrates a scenario run end‑to‑end (plots, CSVs, JSON).

> Works **offline**; if ROS2 is present, add `--publish` to publish to topics:
> `/adaptivecad/fields` (optimizer rows) and `/adaptivecad/redshift_config` (final selection).

## Quickstart

```bash
# 1) Create and activate a venv (optional)
python -m venv .venv && source .venv/bin/activate

# 2) Install minimal deps
python -m pip install numpy matplotlib pillow

# 3) Run a full HOTSPOT scenario (offline)
python adaptivecad_runner.py --scenario hotspot --outdir runs/hotspot_v1

# 4) RIM scenario + publish (if ROS2/Python bridge is installed)
python adaptivecad_runner.py --scenario rim --publish --outdir runs/rim_v1
```

Outputs land under `--outdir`:
- Optimizer CSVs + ΔT plots + GIF (`figs/rho_sweep_panels.gif`)
- Infuser JSON (printed or published) + `redshift_selection.csv`

## Tools

- `adaptivecad_fields_optimizer.py` — 2‑stage coarse→refine ρ search with constraints.
- `adaptivecad_redshift_infuser.py` — qubit‑correlated controller (ρ≈0.25 default + guardrails; tests {0.20, 0.25, 0.30}).
- `adaptivecad_runner.py` — one‑liner to run both with consistent args.

## Scenarios

`hotspot`, `shear_band`, `rim`, `multi_hotspot`, `uniform`

## Notes

- All plots use **matplotlib** only. No external web calls.
- For ROS2 publish, the code tries to import `rclpy` and falls back to printing JSON if missing.
- Tweak constraints with `--density_max` and `--tau_max`.
