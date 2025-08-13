# GU × πₐ Curved 2-Manifold Examples

## Sphere (K>0)
- Icosphere (subdiv=3), R=1.
- Per-vertex Gaussian curvature (angle deficit).
- π_eff(x,r) = π * (1 − (r²/6)K(x)), here r = 0.1R.
- Geodesic circle measured vs theory.

## Saddle (K<0)
- z = (x² − y²)/(2R) on [-a,a]² (R=1, a=1.2).
- Curvature as above; expect π_eff>π near origin for small r.

Run:
```bash
python examples/gu_manifolds/sphere_saddle_examples.py
```
