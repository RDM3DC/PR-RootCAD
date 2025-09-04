# Adaptive Compression Suite (ACS)

[![Sponsor RDM3DC](https://img.shields.io/badge/Sponsor-RDM3DC-ff69b4?logo=github-sponsors)](https://github.com/sponsors/RDM3DC)

A single repo with **three** distinct compression methods we've developed:

1. **ATC — Adaptive Text Compression** (lossless, text):  
   Strips visible spaces/punctuation/case and encodes them in per-character **style bytes** alongside a compact carrier string.

2. **CMC — Curve‑Memory Compression** (event-driven, geometry-aware, typically lossy):  
   Encodes **anchors** at curvature/turning events and reconstructs with an ARP‑style smoother for perceptual fidelity on signals/paths.

3. **GPUC — GPU/Tensor Compression** (numeric arrays):  
   CPU-safe **quantization** and **zero‑suppression**; optional CUDA path via PyTorch if available. Targets inter-device transfer & storage.

---

## Install

```bash
pip install -e .
```

## Command Line (selected)

```bash
# ATC (text)
acs-atc-encode --text "I am in it, okay?  YES!" --out atc.json
acs-atc-decode --in atc.json

# CMC (1D signals)
acs-cmc-encode-1d --in signal.npy --tau 0.02 --max_err 0.01 --out cmc.json
acs-cmc-decode-1d --in cmc.json --n 1000 --out recon.npy

# GPUC (arrays)
acs-gpuc-quantize --in array.npy --out array_q.npz --bits 8
acs-gpuc-dequantize --in array_q.npz --out array_restored.npy

acs-gpuc-zerosuppress --in array.npy --out array_zs.npz --eps 0.0
acs-gpuc-unsuppress --in array_zs.npz --out array_restored.npy
```

> `*.npy/npz` are standard NumPy formats for easy round‑trips without extra deps.
> CUDA is optional; if PyTorch is present, `gpuc` will use GPU tensors transparently.

---

## Repo Layout

```
adaptive-compression-suite/
 ├── atc/      # Adaptive Text Compression (lossless)
 ├── cmc/      # Curve-Memory Compression (event-driven anchors + ARP smoother)
 ├── gpuc/     # GPU/Tensor Compression (quantization + zero suppression, optional CUDA)
 ├── tests/    # pytest
 ├── LICENSE   # MIT
 ├── pyproject.toml
 ├── setup.cfg
 └── .github/workflows/ci.yml
```

---

## Quick Examples

### ATC (text lossless)

```python
from atc.encoder import encode
from atc.decoder import decode

pkg = encode("I am in it, okay?  YES!")
assert decode(pkg) == "I am in it, okay?  YES!"
```

### CMC (1D signals)

```python
import numpy as np
from cmc.one_d import encode_1d, decode_1d

x = np.sin(np.linspace(0, 4*np.pi, 1000))
pkg = encode_1d(x, tau=0.01, max_err=0.005)
y  = decode_1d(pkg, n=len(x), alpha=0.2, mu=0.01)
```

### GPUC (arrays)

```python
import numpy as np
from gpuc.quant import quantize, dequantize

arr = np.random.randn(1024, 1024).astype(np.float32)
pkt = quantize(arr, bits=8)      # -> dict to save as .npz
restored = dequantize(pkt)       # approx reconstruction
```

---

## Notes

- **ATC:** transports JSON with `{ carriers: str, style_b64: base64 }` (1 byte/style per carrier).
- **CMC:** stores anchors (indices/values), optional local slope, and flags; decoder runs ARP‑style smoothing.
- **GPUC:** supports `int8` quantization (per‑tensor or per‑block) and zero‑suppression of near‑zeros; CPU reference implementations included, CUDA optional via PyTorch.
