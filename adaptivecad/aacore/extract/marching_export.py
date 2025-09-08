import numpy as np
from skimage import measure
try:
    from stl import mesh
    _HAVE_STL = True
except Exception:
    _HAVE_STL = False


def sample_sdf(scene, bbox, res):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    zs = np.linspace(zmin, zmax, res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    D = np.empty(X.shape, dtype=np.float32)
    it = np.nditer(D, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        i, j, k = it.multi_index
        D[i, j, k] = scene.sdf((X[i,j,k], Y[i,j,k], Z[i,j,k]))[0]
        it.iternext()
    return D, (xs, ys, zs)


def export_isosurface_to_stl(scene, path, bbox=((-2,-2,-2),(2,2,2)), res=96):
    if not _HAVE_STL:
        raise RuntimeError("numpy-stl not installed; install with `pip install numpy-stl`.")
    D, (xs, ys, zs) = sample_sdf(scene, bbox, res)
    spacing = (xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])
    verts, faces, normals, _ = measure.marching_cubes(D, level=0.0, spacing=spacing)
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        m.vectors[i] = verts[f]
    m.save(path)
    return path
