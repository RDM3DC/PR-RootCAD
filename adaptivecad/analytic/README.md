# AdaptiveCAD Analytic Viewport

The Analytic Viewport provides a triangle-free visualization of 3D models using Signed Distance Fields (SDF) and ray marching. This allows for smooth, mathematically precise rendering without the faceting artifacts common in traditional triangle-based rendering.

## Features

- Triangle-free rendering using ray marching and signed distance fields
- Smooth surfaces with high-quality shading
- Support for PIA (Parametric Interactive Adaptive) geometry modifiers
- Real-time camera navigation and visualization
- Soft shadows and ambient occlusion for enhanced realism
- Gamma correction for accurate color reproduction

## Usage

### Running the Analytic Viewport

You can launch the analytic viewport in one of two ways:

1. From the AdaptiveCAD GUI: Go to View menu > "Open Analytic Viewport (No Triangles)"
2. Directly: Run the `run_analytic_viewport.bat` script or execute `python test_analytic_viewport.py`

### Requirements

- PyQt6 >= 6.9.0
- PyOpenGL >= 3.1.0
- NumPy >= 1.20.0

These dependencies can be installed using:

```bash
pip install -r requirements-pyqt6.txt
```

### Environment Setup

For best compatibility, use Python 3.10 with the provided requirements:

```bash
# Create a Python 3.10 environment (if using conda)
conda create -n adaptivegl python=3.10
conda activate adaptivegl

# Install dependencies
pip install -r requirements-pyqt6.txt
```

## Technical Details

The analytic viewport uses ray marching through signed distance fields (SDFs) to render primitives. This approach offers several advantages:

1. **Perfect Smoothness**: Objects are rendered as mathematical surfaces without discretization
2. **PIA Integration**: Support for parametric adaptive modifications through the PIA beta parameter
3. **Enhanced Visual Quality**: High-quality shading with soft shadows and ambient occlusion
4. **Resolution Independence**: Surface detail is preserved at any zoom level

### Supported Primitives

- Sphere
- Capsule
- Torus

### Implementation

The renderer consists of these key components:

- **Scene Graph**: Data structures for organizing primitives (see `adaptivecad/analytic/scene.py`)
- **GLSL Shaders**: Fragment shader implementing ray marching algorithm (see `adaptivecad/analytic/shaders/`)
- **OpenGL Integration**: PyQt6 QOpenGLWidget wrapper (see `adaptivecad/gui/analytic_viewport.py`)