import numpy as np
try:
    from skimage import measure
    from stl import mesh
except Exception:
    measure = None
    mesh = None

def sample_sdf(scene, bbox, res):
    (xmin,ymin,zmin), (xmax,ymax,zmax) = bbox
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    zs = np.linspace(zmin, zmax, res)
    X,Y,Z = np.meshgrid(xs, ys, zs, indexing='ij')
    D = np.empty(X.shape, dtype=np.float32)
    it = np.nditer(D, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        i,j,k = it.multi_index
        pw = (X[i,j,k], Y[i,j,k], Z[i,j,k])
        it[0] = scene.sdf(pw)[0]
        it.iternext()
    return D, (xs, ys, zs)

def export_isosurface_to_stl(scene, path, bbox=((-2,-2,-2),(2,2,2)), res=180):
    if measure is None or mesh is None:
        raise ImportError("Requires 'scikit-image' and 'numpy-stl' (pip install scikit-image numpy-stl)")
    D, (xs,ys,zs) = sample_sdf(scene, bbox, res)
    spacing = (xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])
    verts, faces, _normals, _ = measure.marching_cubes(D, level=0.0, spacing=spacing)
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        m.vectors[i] = verts[f]
    m.save(path)
    return path
