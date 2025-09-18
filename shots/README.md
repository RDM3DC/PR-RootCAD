Hero Shot Renderer (Quasi‑Crystal + Superellipsoid)

Quick start (Windows PowerShell):

1) Render 10s @ 1080p30 to PNG frames

```powershell
python shots\render_quasicrystal_hero.py --w 1920 --h 1080 --fps 30 --seconds 10 --out renders\hero_qc
```

2) Encode MP4 with ffmpeg

```powershell
ffmpeg -y -framerate 30 -i renders/hero_qc/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow renders/hero_qc.mp4
```

Notes
- Uses analytic SDFs only; no tessellation.
- Superellipsoid shell is modeled as outer solid minus inner subtract (thickness ~0.10).
- Quasi‑crystal field parameters: [scale, iso, thickness]; animation scales from ~2→6 over the shot.
- Tweak palette via `scene.bg_color` and `scene.env_light` in the script.
