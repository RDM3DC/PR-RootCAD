import bpy, csv, os
from mathutils import Vector

def load_vertices(path):
    pts=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])) )
    return pts

def load_edges(path):
    E=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            E.append((int(row["i"]), int(row["j"])) )
    return E

def build_mesh(pts, edges, name="Tesseract3D"):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(pts, edges, [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    return obj

base = os.path.dirname(__file__)
pts = load_vertices(os.path.join(base, "out", "vertices.csv"))
edges = load_edges(os.path.join(base, "out", "edges.csv"))
build_mesh(pts, edges)
print("[ok] imported tesseract to Blender")
