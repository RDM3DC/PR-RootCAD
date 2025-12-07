# blender_addons/adaptivecad_pia_aniso.py
bl_info = {
    "name": "AdaptiveCAD Anisotropic Distance (FMM-lite)",
    "author": "AdaptiveCAD",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > AdaptiveCAD",
    "description": "Compute a small anisotropic distance map and print samples",
    "category": "3D View",
}

import bpy, math
from bpy.props import FloatProperty

try:
    import numpy as np
    from adaptive_pi.aniso_fmm import anisotropic_fmm, metric_const_aniso
except Exception:
    np = None


class ADAP_OP_AnisoDistance(bpy.types.Operator):
    bl_idname = "adaptivecad.aniso_distance"
    bl_label = "Run Aniso Distance"
    a: FloatProperty(name="a (G11)", default=1.3, min=0.01, max=100.0)
    b: FloatProperty(name="b (G22)", default=1.0, min=0.01, max=100.0)

    def execute(self, ctx):
        if np is None:
            self.report({"ERROR"}, "NumPy/adaptive_pi not available")
            return {"CANCELLED"}
        nx = ny = 129
        G = metric_const_aniso(nx, ny, a=self.a, b=self.b)
        src = (nx // 2, ny // 2)
        T = anisotropic_fmm(G, src, use_diagonals=True)
        mid = src[1]
        xs = [0, 5, 10, 20, 30, 40]
        row_x = [round(float(T[src[0] + k, mid]), 4) for k in xs]
        self.report({"INFO"}, f"T(+x)={row_x}")
        return {"FINISHED"}


class ADAP_PT_AnisoDistance(bpy.types.Panel):
    bl_label = "Anisotropic Distance"
    bl_idname = "ADAPTIVECAD_PT_ANISO_DISTANCE"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AdaptiveCAD"

    def draw(self, ctx):
        col = self.layout.column(align=True)
        op = col.operator(ADAP_OP_AnisoDistance.bl_idname, text="Run Demo")
        col.prop(op, "a")
        col.prop(op, "b")


classes = (ADAP_OP_AnisoDistance, ADAP_PT_AnisoDistance)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
