bl_info = {
    "name": "AdaptiveCAD πₐ Toolpath",
    "author": "AdaptiveCAD",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Add > Curve > πₐ Toolpath",
    "description": "Generate πₐ-adaptive circle path as a Curve and a Euclidean reference",
    "category": "Add Curve",
}

import bpy, math
from math import pi
from mathutils import Vector


def trC_xy(x, y):
    from math import exp

    g1 = exp(-((x - 15) ** 2 + (y - 5) ** 2) / (2 * 8**2))
    g2 = -0.8 * exp(-((x + 10) ** 2 + (y + 12) ** 2) / (2 * 10**2))
    return 0.6 * g1 + 0.6 * g2


def generate_points(R0=30.0, N=720, lam_field=0.04):
    ds = 2 * pi * R0 / N
    theta = 0.0
    pts = []
    for k in range(N):
        x = R0 * math.cos(theta)
        y = R0 * math.sin(theta)
        pts.append((x, y))
        pia_loc = pi * (1 + lam_field * trC_xy(x, y))
        dtheta = (pi / pia_loc) * (ds / R0)
        theta += dtheta
    pts.append(pts[0])
    return pts


def make_curve(name, pts, closed=True):
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new(type="POLY")
    spline.points.add(len(pts) - 1)
    for i, (x, y) in enumerate(pts):
        spline.points[i].co = (x, y, 0.0, 1.0)
    spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return obj


class ADAPTIVECAD_OT_pia_toolpath(bpy.types.Operator):
    bl_idname = "adaptivecad.pia_toolpath"
    bl_label = "Add πₐ Toolpath"
    bl_options = {"REGISTER", "UNDO"}

    radius: bpy.props.FloatProperty(name="Radius (mm)", default=30.0, min=1.0, soft_max=200.0)
    steps: bpy.props.IntProperty(name="Steps", default=720, min=60, soft_max=2000)
    lam: bpy.props.FloatProperty(name="λ (coupling)", default=0.04, min=0.0, soft_max=0.2)

    def execute(self, context):
        pts = generate_points(self.radius, self.steps, self.lam)
        obj_pia = make_curve("piA_Path", pts, closed=True)
        # Euclidean reference circle
        ref = [
            (self.radius * math.cos(t), self.radius * math.sin(t))
            for t in [2 * pi * i / 256 for i in range(257)]
        ]
        obj_ref = make_curve("Euclid_Circle", ref, closed=True)
        # small bevel for visibility
        for obj in (obj_pia, obj_ref):
            obj.data.bevel_depth = 0.2
        return {"FINISHED"}


def menu_func(self, context):
    self.layout.operator(
        ADAPTIVECAD_OT_pia_toolpath.bl_idname, text="πₐ Toolpath", icon="CURVE_DATA"
    )


def register():
    bpy.utils.register_class(ADAPTIVECAD_OT_pia_toolpath)
    bpy.types.VIEW3D_MT_curve_add.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_curve_add.remove(menu_func)
    bpy.utils.unregister_class(ADAPTIVECAD_OT_pia_toolpath)
