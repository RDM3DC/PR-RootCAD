# adaptivecad/gui/analytic_viewport.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent
from PyQt6.QtCore import Qt, QSize
from OpenGL.GL import *
import numpy as np
from pathlib import Path
from adaptivecad.aacore.sdf import (
    Scene as AACoreScene, Prim, KIND_SPHERE, KIND_BOX, KIND_CAPSULE, KIND_TORUS,
    OP_SOLID, OP_SUBTRACT, MAX_PRIMS
)
from PyQt6.QtWidgets import (
    QWidget as _QWidgetAlias, QVBoxLayout as _QVBoxLayoutAlias, QHBoxLayout as _QHBoxLayoutAlias,
    QLabel as _QLabelAlias, QComboBox, QCheckBox, QPushButton, QSlider, QGroupBox, QFormLayout, QSpinBox
)
import time, os
import json
try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

def load_text(p):
    return Path(p).read_text(encoding="utf-8")

class AnalyticViewport(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3,3); fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSamples(4)  # 4x MSAA
        self.setFormat(fmt)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Shared AACore scene (will be injected later or created fresh)
        self.scene = AACoreScene()
        # camera
        self.cam_target = np.array([0.0,0.0,0.0], np.float32)
        self.distance = 6.0
        self.yaw=0.0; self.pitch=0.0
        self.cam_pos = np.array([0.0,0.0,self.distance], np.float32)
        self._last_mouse = None
        self._last_buttons = Qt.MouseButton.NoButton
        # shader handles
        self.prog = None
        # debug / feature toggles (could later expose via UI)
        self.debug_mode = 0          # 0 beauty
        self.use_analytic_aa = 1
        self.use_toon = 0
        self.toon_levels = 4.0
        self.curv_strength = 1.0
        self.use_foveated = 1
        self.far_plane = 200.0
        self.show_beta_overlay = 0
        self.beta_overlay_intensity = 0.65
        self.beta_scale = 1.0
        self.beta_cmap = 0  # 0 legacy,1 viridis,2 plasma,3 diverge
        self._settings_path = os.path.join(os.path.expanduser('~'), '.adaptivecad_analytic.json')
        self._load_settings()
        self._beta_lut_tex = None
        self._curv_lut_tex = None
        self.use_curv_lut = 1
        # listen for scene edits (debounced via timer-less simple flag)
        self._scene_dirty = True
        self.scene.on_changed(self._on_scene_changed)
        # build a tiny demo scene (aacore prims)
        self._demo_scene()

    def _on_scene_changed(self):
        self._scene_dirty = True
        self.update()

    def _demo_scene(self):
        self.scene.add(Prim(KIND_SPHERE, [1.2,0,0,0], beta=0.12, color=(0.9,0.4,0.3)))
        self.scene.add(Prim(KIND_CAPSULE, [0.4,2.0,0,0], beta=0.0, color=(0.3,0.7,0.9)))
        self.scene.add(Prim(KIND_TORUS, [1.2,0.25,0,0], beta=0.05, color=(0.6,0.9,0.5)))

    def sizeHint(self): return QSize(800,600)

    # --- OpenGL lifecycle ---
    def initializeGL(self):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)  # Enable MSAA
        shader_dir = Path(__file__).parent.parent / "analytic" / "shaders"
        vs = load_text(shader_dir / "sdf.vert")
        fs = load_text(shader_dir / "sdf.frag")
        self.prog = glCreateProgram()
        vsh = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vsh, vs); glCompileShader(vsh)
        if not glGetShaderiv(vsh, GL_COMPILE_STATUS): raise RuntimeError(glGetShaderInfoLog(vsh).decode())
        fsh = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fsh, fs); glCompileShader(fsh)
        if not glGetShaderiv(fsh, GL_COMPILE_STATUS): raise RuntimeError(glGetShaderInfoLog(fsh).decode())
        glAttachShader(self.prog, vsh); glAttachShader(self.prog, fsh); glLinkProgram(self.prog)
        if not glGetProgramiv(self.prog, GL_LINK_STATUS): raise RuntimeError(glGetProgramInfoLog(self.prog).decode())
        glDeleteShader(vsh); glDeleteShader(fsh)
        # fullscreen quad
        self._vao = glGenVertexArrays(1); glBindVertexArray(self._vao)
        v = np.array([-1,-1,  1,-1, -1, 1,   1,1], dtype=np.float32)
        self._vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    def _cam_basis(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        # forward in -Z
        f = np.array([sy*cp, sp, -cy*cp], np.float32)
        r = np.array([ cy, 0.0,  sy], np.float32)
        u = np.cross(r, f)
        R = np.stack([r,u,-f], axis=1).astype(np.float32)  # columns
        return R

    def _update_camera(self):
        # forward vector points from cam to target, so place camera at target - f*distance
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        f = np.array([sy*cp, sp, -cy*cp], np.float32)
        self.cam_pos = self.cam_target - f * self.distance
        self.update()

    def paintGL(self):
        self._draw_frame(debug_override=None)

    def _draw_frame(self, debug_override=None):
        glViewport(0,0,self.width(), self.height())
        glClearColor(*self.scene.bg_color, 1.0); glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.prog)
        # uniforms
        glUniform3f(glGetUniformLocation(self.prog,"u_cam_pos"), *self.cam_pos)
        R = self._cam_basis(); glUniformMatrix3fv(glGetUniformLocation(self.prog,"u_cam_rot"), 1, GL_FALSE, R.T)
        glUniform2f(glGetUniformLocation(self.prog,"u_res"), float(self.width()), float(self.height()))
        glUniform3f(glGetUniformLocation(self.prog,"u_env"), *self.scene.env_light)
        glUniform3f(glGetUniformLocation(self.prog,"u_bg"), *self.scene.bg_color)
        # upload scene
        pack = self.scene.to_gpu_structs(max_prims=MAX_PRIMS)
        n = int(pack['count'])
        glUniform1i(glGetUniformLocation(self.prog,"u_count"), n)
        def U(name): return glGetUniformLocation(self.prog, name)
        glUniform1iv(U("u_kind"), n, pack['kind'])
        glUniform1iv(U("u_op"),   n, pack['op'])
        glUniform1fv(U("u_beta"), n, pack['beta'])
        glUniform3fv(U("u_color"), n, pack['color'])
        glUniform4fv(U("u_params"), n, pack['params'])
        glUniformMatrix4fv(U("u_xform"), n, GL_FALSE, pack['xform'])
        # draw (beauty)
        glUniform1i(glGetUniformLocation(self.prog,"u_mode"), 0)
        # new uniforms
        mode_val = self.debug_mode if debug_override is None else debug_override
        glUniform1i(glGetUniformLocation(self.prog, "u_debug"), int(mode_val))
        glUniform1f(glGetUniformLocation(self.prog, "u_far"), float(self.far_plane))
        glUniform1i(glGetUniformLocation(self.prog, "u_use_analytic_aa"), int(self.use_analytic_aa))
        glUniform1i(glGetUniformLocation(self.prog, "u_use_toon"), int(self.use_toon))
        glUniform1f(glGetUniformLocation(self.prog, "u_toon_levels"), float(self.toon_levels))
        glUniform1f(glGetUniformLocation(self.prog, "u_curv_strength"), float(self.curv_strength))
        glUniform1i(glGetUniformLocation(self.prog, "u_use_foveated"), int(self.use_foveated))
        glUniform1i(glGetUniformLocation(self.prog, "u_show_beta_overlay"), int(self.show_beta_overlay))
        glUniform1f(glGetUniformLocation(self.prog, "u_beta_overlay_intensity"), float(self.beta_overlay_intensity))
        glUniform1f(glGetUniformLocation(self.prog, "u_beta_scale"), float(self.beta_scale))
        glUniform1i(glGetUniformLocation(self.prog, "u_beta_cmap"), int(self.beta_cmap))
        glBindVertexArray(self._vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def keyPressEvent(self, event):
        k = event.key()
        changed = False
        # number keys for debug modes
        if k in (Qt.Key_0, Qt.Key.Key_0): self.debug_mode=0; changed=True
        elif k in (Qt.Key_1, Qt.Key.Key_1): self.debug_mode=1; changed=True
        elif k in (Qt.Key_2, Qt.Key.Key_2): self.debug_mode=2; changed=True
        elif k in (Qt.Key_3, Qt.Key.Key_3): self.debug_mode=3; changed=True
        elif k in (Qt.Key_4, Qt.Key.Key_4): self.debug_mode=4; changed=True
        elif k == Qt.Key_Space:
            self.debug_mode = (self.debug_mode + 1) % 5; changed=True
        elif k == Qt.Key_A: self.use_analytic_aa ^=1; changed=True
        elif k == Qt.Key_F: self.use_foveated ^=1; changed=True
        elif k == Qt.Key_T: self.use_toon ^=1; changed=True
        elif k == Qt.Key_Plus or k == Qt.Key_Equal: # '+' (with shift) or '='
            self.toon_levels = min(self.toon_levels+1.0, 8.0); changed=True
        elif k == Qt.Key_Minus: self.toon_levels = max(self.toon_levels-1.0, 1.0); changed=True
        elif k == Qt.Key_C:
            # cycle curvature strength
            if self.curv_strength < 0.25: self.curv_strength = 0.5
            elif self.curv_strength < 0.75: self.curv_strength = 1.0
            else: self.curv_strength = 0.0
            changed=True
        elif k == Qt.Key_G: self.save_gbuffers(); return
        elif k == Qt.Key_H:
            print(self._shortcuts_help())
        elif k == Qt.Key_B:
            self.show_beta_overlay ^=1; changed=True
        elif k == Qt.Key_Period: # increase beta overlay intensity
            self.beta_overlay_intensity = min(1.0, self.beta_overlay_intensity+0.05); changed=True
        elif k == Qt.Key_Comma:  # decrease intensity
            self.beta_overlay_intensity = max(0.0, self.beta_overlay_intensity-0.05); changed=True
        elif k == Qt.Key_Slash:  # increase beta scale
            self.beta_scale = min(8.0, self.beta_scale * 1.25); changed=True
        elif k == Qt.Key_Question: # SHIFT + '/' may map same; ensure alternative
            self.beta_scale = min(8.0, self.beta_scale * 1.25); changed=True
        elif k == Qt.Key_Backslash: # decrease beta scale
            self.beta_scale = max(0.125, self.beta_scale / 1.25); changed=True
        elif k == Qt.Key_V: # cycle beta color map
            self.beta_cmap = (self.beta_cmap + 1) % 4; changed=True
        if changed:
            self._update_title(); self.update(); self._save_settings()
        else:
            super().keyPressEvent(event)

    # --- Mouse Interaction ---
    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse = event.position()
        self._last_buttons = event.buttons()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None:
            self._last_mouse = event.position()
        dx = event.position().x() - self._last_mouse.x()
        dy = event.position().y() - self._last_mouse.y()
        buttons = event.buttons() or self._last_buttons
        updated = False
        # Orbit (LMB)
        if buttons & Qt.MouseButton.LeftButton:
            sens = 0.005
            self.yaw   += dx * sens
            self.pitch += dy * sens
            # clamp pitch to avoid gimbal flip
            self.pitch = float(np.clip(self.pitch, -1.45, 1.45))
            self._update_camera()
            updated = True
        # Pan (RMB)
        if buttons & Qt.MouseButton.RightButton:
            # derive right/up from current basis
            R = self._cam_basis()
            right = R[:,0]; up = R[:,1]
            pan_sens = self.distance * 0.0015
            self.cam_target -= right * dx * pan_sens
            self.cam_target += up * dy * pan_sens
            self._update_camera()
            updated = True
        self._last_mouse = event.position()
        if not updated:
            super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y() / 120.0  # 1 per notch
        zoom_factor = 1.0 - delta * 0.1
        zoom_factor = max(0.1, min(2.0, zoom_factor))
        self.distance *= zoom_factor
        self.distance = float(np.clip(self.distance, 0.5, 200.0))
        self._update_camera()
        super().wheelEvent(event)

    def _update_title(self):
        try:
            mode_names = ['beauty','normals','id','depth','thickness']
            m = mode_names[self.debug_mode] if 0<=self.debug_mode < len(mode_names) else str(self.debug_mode)
            cmap_names = ['legacy','viridis','plasma','diverge']
            cm = cmap_names[self.beta_cmap]
            self.window().setWindowTitle(
                f"Analytic Viewport - {m} | AA={'on' if self.use_analytic_aa else 'off'} Toon={'on' if self.use_toon else 'off'} Lvl={int(self.toon_levels)} Curv={self.curv_strength:.1f} FovStep={'on' if self.use_foveated else 'off'} β={'on' if self.show_beta_overlay else 'off'} βI={self.beta_overlay_intensity:.2f} βS={self.beta_scale:.2f} {cm}"
            )
        except Exception:
            pass

    def _shortcuts_help(self):
        return (
            "Shortcuts:\n"
            "0..4 : debug modes (beauty,normals,id,depth,thickness)\n"
            "Space : cycle debug mode\n"
            "A : toggle analytic AA\n"
            "F : toggle foveated steps\n"
            "T : toggle toon\n"
            "+ / - : toon levels up/down\n"
            "C : cycle curvature strength (0,0.5,1.0,0)\n"
            "G : save G-buffers (beauty + normals/id/depth/thickness)\n"
            "H : print this help"
        )

    def save_gbuffers(self):
        # Ensure context current
        self.makeCurrent()
        out_dir = os.path.join(os.getcwd(), 'gbuffers')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        targets = [(-1,'beauty'), (1,'normals'), (2,'id'), (3,'depth'), (4,'thickness')]
        w,h = self.width(), self.height()
        prev_mode = self.debug_mode
        # Render beauty first (override debug to 0 by setting debug_override=None with debug_mode=0)
        for mode,label in targets:
            if mode==-1:
                # beauty
                self.debug_mode=0
                self._draw_frame(debug_override=0)
            else:
                self._draw_frame(debug_override=mode)
            glFinish()
            data = glReadPixels(0,0,w,h, GL_RGBA, GL_UNSIGNED_BYTE)
            import numpy as _np
            img = _np.frombuffer(data, dtype=_np.uint8).reshape(h, w, 4)
            img = img[::-1,:,:]  # flip vertically
            path = os.path.join(out_dir, f"{timestamp}_{label}.png")
            if _HAVE_PIL:
                Image.fromarray(img, 'RGBA').save(path)
            else:
                # fallback raw dump
                img.tofile(path + '.raw')
            print(f"Saved {label} -> {path}")
        self.debug_mode = prev_mode
        self.update()


class AnalyticViewportPanel(_QWidgetAlias):
    """Composite widget: analytic SDF viewport + control panel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = AnalyticViewport(self)
        self._build_ui()
        self.setWindowTitle("Analytic Viewport (Panel)")
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _build_ui(self):
        root = _QHBoxLayoutAlias(self)
        root.setContentsMargins(4,4,4,4)
        root.addWidget(self.view, 1)

        side = _QVBoxLayoutAlias(); side.setSpacing(6)

        # --- Debug Modes ---
        gb_modes = QGroupBox("Debug / Modes"); fm = QFormLayout(); gb_modes.setLayout(fm)
        self.mode_box = QComboBox(); self.mode_box.addItems(["Beauty","Normals","ID","Depth","Thickness"])
        self.mode_box.currentIndexChanged.connect(self._on_mode); fm.addRow("Mode", self.mode_box)
        side.addWidget(gb_modes)

        # --- Feature Toggles ---
        gb_feat = QGroupBox("Features"); vf = QVBoxLayout(); gb_feat.setLayout(vf)
        self.cb_aa = QCheckBox("Analytic AA"); self.cb_aa.setChecked(True); self.cb_aa.stateChanged.connect(self._on_toggle)
        self.cb_fov = QCheckBox("Foveated Steps"); self.cb_fov.setChecked(True); self.cb_fov.stateChanged.connect(self._on_toggle)
        self.cb_toon = QCheckBox("Toon"); self.cb_toon.setChecked(False); self.cb_toon.stateChanged.connect(self._on_toggle)
        for w in (self.cb_aa, self.cb_fov, self.cb_toon): vf.addWidget(w)
        side.addWidget(gb_feat)

        # --- Parameters ---
        gb_sliders = QGroupBox("Parameters"); fs = QFormLayout(); gb_sliders.setLayout(fs)
        self.slider_toon = QSlider(Qt.Orientation.Horizontal); self.slider_toon.setRange(1,8); self.slider_toon.setValue(4); self.slider_toon.valueChanged.connect(self._on_toon_levels)
        self.slider_curv = QSlider(Qt.Orientation.Horizontal); self.slider_curv.setRange(0,100); self.slider_curv.setValue(100); self.slider_curv.valueChanged.connect(self._on_curv)
        self.cb_beta = QCheckBox("β Overlay"); self.cb_beta.setChecked(False); self.cb_beta.stateChanged.connect(self._on_toggle)
        self.slider_beta_int = QSlider(Qt.Orientation.Horizontal); self.slider_beta_int.setRange(0,100); self.slider_beta_int.setValue(int(self.view.beta_overlay_intensity*100)); self.slider_beta_int.valueChanged.connect(self._on_beta_int)
        self.slider_beta_scale = QSlider(Qt.Orientation.Horizontal); self.slider_beta_scale.setRange(1,800); self.slider_beta_scale.setValue(int(self.view.beta_scale*100)); self.slider_beta_scale.valueChanged.connect(self._on_beta_scale)
        self.cmap_box = QComboBox(); self.cmap_box.addItems(["Legacy","Viridis","Plasma","Diverge"]); self.cmap_box.currentIndexChanged.connect(self._on_cmap)
        fs.addRow("Toon Levels", self.slider_toon)
        fs.addRow("Curvature", self.slider_curv)
        fs.addRow(self.cb_beta)
        fs.addRow("β Intensity", self.slider_beta_int)
        fs.addRow("β Scale", self.slider_beta_scale)
        fs.addRow("β Colormap", self.cmap_box)
        side.addWidget(gb_sliders)

        # --- Actions ---
        gb_act = QGroupBox("Actions"); va = QVBoxLayout(); gb_act.setLayout(va)
        btn_save = QPushButton("Save G-Buffers [G]"); btn_save.setEnabled(False)
        btn_help = QPushButton("Print Shortcuts [H]"); btn_help.clicked.connect(lambda: print(self.view._shortcuts_help()))
        va.addWidget(btn_save); va.addWidget(btn_help); side.addWidget(gb_act)

        # --- Environment ---
        gb_env = QGroupBox("Environment"); fe = QFormLayout(); gb_env.setLayout(fe)
        self._bg_sliders = []
        for i, ch in enumerate(['R','G','B']):
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(0,255); s.setValue(int(self.view.scene.bg_color[i]*255)); s.valueChanged.connect(self._on_bg_changed)
            fe.addRow(f"BG {ch}", s); self._bg_sliders.append(s)
        self._env_sliders = []
        for i, ch in enumerate(['X','Y','Z']):
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(-200,200); s.setValue(int(self.view.scene.env_light[i]*100)); s.valueChanged.connect(self._on_env_changed)
            fe.addRow(f"Light {ch}", s); self._env_sliders.append(s)
        side.addWidget(gb_env)

        # --- Primitives + Editing ---
        gb_prims = QGroupBox("Primitives"); vp = QVBoxLayout(); gb_prims.setLayout(vp)
        self._prim_list_label = QLabel("(0)")
        self._prim_buttons_box = QVBoxLayout()
        btn_add_sphere = QPushButton("Add Sphere"); btn_add_sphere.clicked.connect(lambda: self._add_prim('sphere'))
        btn_add_box = QPushButton("Add Box"); btn_add_box.clicked.connect(lambda: self._add_prim('box'))
        btn_add_capsule = QPushButton("Add Capsule"); btn_add_capsule.clicked.connect(lambda: self._add_prim('capsule'))
        btn_add_torus = QPushButton("Add Torus"); btn_add_torus.clicked.connect(lambda: self._add_prim('torus'))
        btn_del_last = QPushButton("Delete Last"); btn_del_last.clicked.connect(self._del_last)
        for b in (self._prim_list_label, btn_add_sphere, btn_add_box, btn_add_capsule, btn_add_torus, btn_del_last):
            vp.addWidget(b)
        vb_holder = QWidget(); vb_holder.setLayout(self._prim_buttons_box); vp.addWidget(vb_holder)
        # Edit group
        self._edit_group = QGroupBox("Edit Selected"); fe2 = QFormLayout(); self._edit_group.setLayout(fe2)
        from PyQt6.QtWidgets import QDoubleSpinBox, QComboBox as _QCB2, QColorDialog
        def dspin(r=(-10,10), step=0.01, val=0.0):
            sp = QDoubleSpinBox(); sp.setRange(r[0], r[1]); sp.setSingleStep(step); sp.setDecimals(4); sp.setValue(val); return sp
        self._sp_pos = [dspin() for _ in range(3)]
        self._sp_param = [dspin((0,10),0.01,0.5) for _ in range(2)]
        self._sp_beta = dspin((-1,1),0.01,0.0)
        self._btn_color = QPushButton("Color…")
        self._op_box = _QCB2(); self._op_box.addItems(["solid","subtract"])
        fe2.addRow("Pos X/Y/Z", self._make_row(self._sp_pos))
        fe2.addRow("Param A/B", self._make_row(self._sp_param))
        fe2.addRow("Beta", self._sp_beta)
        fe2.addRow("Color", self._btn_color)
        fe2.addRow("Op", self._op_box)
        self._edit_group.setEnabled(False); vp.addWidget(self._edit_group)
        for sp in self._sp_pos: sp.valueChanged.connect(self._apply_edit)
        for sp in self._sp_param: sp.valueChanged.connect(self._apply_edit)
        self._sp_beta.valueChanged.connect(self._apply_edit); self._op_box.currentIndexChanged.connect(self._apply_edit)
        def pick_color():
            from PyQt6.QtGui import QColor
            c = QColorDialog.getColor(QColor(200,180,160), self, "Primitive Color")
            if c.isValid():
                self._current_color = (c.red()/255.0, c.green()/255.0, c.blue()/255.0)
                self._apply_edit()
        self._btn_color.clicked.connect(pick_color)
        self._current_sel = -1; self._current_color = (0.8,0.7,0.6)
        side.addWidget(gb_prims)

        # finalize
        side.addStretch(1); root.addLayout(side)
        self._refresh_prim_label()

    # --- UI callbacks ---
    def _on_mode(self, idx:int):
        self.view.debug_mode = idx
        self.view._update_title(); self.view.update()

    def _on_toggle(self):
        self.view.use_analytic_aa = 1 if self.cb_aa.isChecked() else 0
        self.view.use_foveated = 1 if self.cb_fov.isChecked() else 0
        self.view.use_toon = 1 if self.cb_toon.isChecked() else 0
        self.view.show_beta_overlay = 1 if self.cb_beta.isChecked() else 0
        self.view._update_title(); self.view.update(); self.view._save_settings()

    def _on_toon_levels(self, v:int):
        self.view.toon_levels = float(v)
        if self.view.use_toon:
            self.view.update(); self.view._save_settings()

    def _on_curv(self, v:int):
        self.view.curv_strength = v/100.0
        self.view.update(); self.view._save_settings()

    def _on_beta_int(self, v:int):
        self.view.beta_overlay_intensity = v/100.0
        if self.view.show_beta_overlay:
            self.view.update(); self.view._save_settings()

    def _on_beta_scale(self, v:int):
        self.view.beta_scale = v/100.0
        if self.view.show_beta_overlay:
            self.view.update(); self.view._save_settings()

    def _on_cmap(self, idx:int):
        self.view.beta_cmap = idx
        if self.view.show_beta_overlay:
            self.view.update()
        self.view._save_settings(); self.view._update_title()

    # --- Environment callbacks ---
    def _on_bg_changed(self, _v):
        for i,s in enumerate(self._bg_sliders):
            self.view.scene.bg_color[i] = s.value()/255.0
        self.view.update()

    def _on_env_changed(self, _v):
        for i,s in enumerate(self._env_sliders):
            self.view.scene.env_light[i] = s.value()/100.0
        self.view.update()

    # --- Primitive management ---
    def _add_prim(self, kind:str):
        if len(self.view.scene.prims) >= MAX_PRIMS:
            print(f"[prims] Max {MAX_PRIMS} reached")
            return
        if kind=='sphere':
            self.view.scene.add(Prim(KIND_SPHERE, [0.6,0,0,0], beta=0.05, color=(0.9,0.5,0.4)))
        elif kind=='box':
            self.view.scene.add(Prim(KIND_BOX, [0.5,0.5,0.5,0], beta=0.0, color=(0.4,0.7,0.9)))
        elif kind=='capsule':
            self.view.scene.add(Prim(KIND_CAPSULE, [0.3,1.5,0,0], beta=0.02, color=(0.6,0.9,0.5)))
        elif kind=='torus':
            self.view.scene.add(Prim(KIND_TORUS, [0.9,0.25,0,0], beta=0.03, color=(0.8,0.6,0.4)))
        self._refresh_prim_label(); self.view.update()

    def _del_last(self):
        if self.view.scene.prims:
            self.view.scene.remove_index(len(self.view.scene.prims)-1)
            self._refresh_prim_label(); self.view.update()

    def _refresh_prim_label(self):
        self._prim_list_label.setText(f"({len(self.view.scene.prims)})")
        # rebuild primitive buttons
        while self._prim_buttons_box.count():
            it = self._prim_buttons_box.takeAt(0)
            w = it.widget()
            if w: w.deleteLater()
        for idx, pr in enumerate(self.view.scene.prims):
            b = QPushButton(f"#{idx} {self._kind_name(pr.kind)}")
            b.setCheckable(True)
            if idx == self._current_sel: b.setChecked(True)
            def make_cb(i):
                return lambda _=False: self._select_prim(i)
            b.clicked.connect(make_cb(idx))
            self._prim_buttons_box.addWidget(b)
        self._prim_buttons_box.addStretch(1)

    def _kind_name(self, k):
        return {KIND_SPHERE:"Sphere",KIND_BOX:"Box",KIND_CAPSULE:"Capsule",KIND_TORUS:"Torus"}.get(k,str(k))

    def _make_row(self, widgets):
        box = QWidget(); hl = QHBoxLayout(box); hl.setContentsMargins(0,0,0,0)
        for w in widgets: hl.addWidget(w)
        return box

    def _select_prim(self, idx:int):
        if not (0 <= idx < len(self.view.scene.prims)):
            self._current_sel = -1; self._edit_group.setEnabled(False); return
        self._current_sel = idx
        pr = self.view.scene.prims[idx]
        # position from transform matrix (assume affine last column xyz)
        pos = pr.xform.M[:3,3]
        for i in range(3): self._sp_pos[i].blockSignals(True); self._sp_pos[i].setValue(float(pos[i])); self._sp_pos[i].blockSignals(False)
        # params
        self._sp_param[0].blockSignals(True); self._sp_param[0].setValue(float(pr.params[0])); self._sp_param[0].blockSignals(False)
        self._sp_param[1].blockSignals(True); self._sp_param[1].setValue(float(pr.params[1] if pr.kind!=KIND_SPHERE else 0.0)); self._sp_param[1].blockSignals(False)
        self._sp_beta.blockSignals(True); self._sp_beta.setValue(pr.beta); self._sp_beta.blockSignals(False)
        self._op_box.blockSignals(True); self._op_box.setCurrentIndex(0 if pr.op=='solid' else 1); self._op_box.blockSignals(False)
        self._current_color = tuple(pr.color[:3])
        self._edit_group.setEnabled(True)

    def _apply_edit(self):
        if self._current_sel < 0 or self._current_sel >= len(self.view.scene.prims):
            return
        pr = self.view.scene.prims[self._current_sel]
        # apply position
        M = pr.xform.M.copy()
        for i in range(3): M[i,3] = self._sp_pos[i].value()
        pr.xform.M = M
        # apply params
        pr.params[0] = self._sp_param[0].value()
        if pr.kind != KIND_SPHERE:
            pr.params[1] = self._sp_param[1].value()
        pr.beta = self._sp_beta.value()
        pr.op = 'solid' if self._op_box.currentIndex()==0 else 'subtract'
        pr.color[:3] = np.array(self._current_color[:3])
        # notify & refresh
        self.view.scene._notify()
        self.view.update()

    # Convenience passthroughs
    def viewport(self) -> AnalyticViewport: return self.view

def create_analytic_viewport_with_panel(parent=None):
    return AnalyticViewportPanel(parent)

# --- persistence helpers on AnalyticViewport ---
def _analytic_settings_defaults():
    return {
        'debug_mode':0,
        'use_analytic_aa':1,
        'use_toon':0,
        'toon_levels':4.0,
        'curv_strength':1.0,
        'use_foveated':1,
        'show_beta_overlay':0,
        'beta_overlay_intensity':0.65,
        'beta_scale':1.0,
        'beta_cmap':0
    }

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _apply_settings(self, data):
    self.debug_mode = int(data.get('debug_mode',0))
    self.use_analytic_aa = int(data.get('use_analytic_aa',1))
    self.use_toon = int(data.get('use_toon',0))
    self.toon_levels = float(_clamp(data.get('toon_levels',4.0),1.0,16.0))
    self.curv_strength = float(_clamp(data.get('curv_strength',1.0),0.0,2.0))
    self.use_foveated = int(data.get('use_foveated',1))
    self.show_beta_overlay = int(data.get('show_beta_overlay',0))
    self.beta_overlay_intensity = float(_clamp(data.get('beta_overlay_intensity',0.65),0.0,1.0))
    self.beta_scale = float(_clamp(data.get('beta_scale',1.0),0.01,32.0))
    self.beta_cmap = int(_clamp(data.get('beta_cmap',0),0,3))

def _load_settings(self):
    try:
        if os.path.exists(self._settings_path):
            with open(self._settings_path,'r',encoding='utf-8') as f:
                data=json.load(f)
        else:
            data=_analytic_settings_defaults()
        _apply_settings(self, data)
    except Exception as e:
        print(f"(analytic settings) load failed: {e}")

def _save_settings(self):
    try:
        data={
            'debug_mode': self.debug_mode,
            'use_analytic_aa': self.use_analytic_aa,
            'use_toon': self.use_toon,
            'toon_levels': self.toon_levels,
            'curv_strength': self.curv_strength,
            'use_foveated': self.use_foveated,
            'show_beta_overlay': self.show_beta_overlay,
            'beta_overlay_intensity': self.beta_overlay_intensity,
            'beta_scale': self.beta_scale,
            'beta_cmap': self.beta_cmap
        }
        with open(self._settings_path,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
    except Exception as e:
        print(f"(analytic settings) save failed: {e}")

# Monkey-patch methods onto AnalyticViewport class after definition (simple, avoids editing earlier class body heavily)
AnalyticViewport._load_settings = _load_settings
AnalyticViewport._save_settings = _save_settings
