"""Analytic SDF OpenGL viewport & control panel (PySide6 version).

Refactored to standardize on PySide6 (no PyQt6 mixing) and allow
injection of a shared AACore analytic scene so multiple view components
can operate on the same underlying primitive set.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QComboBox, QCheckBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent
from PySide6.QtCore import Qt, QSize
from OpenGL.GL import *
import numpy as np
from pathlib import Path
from adaptivecad.aacore.sdf import (
    Scene as AACoreScene, Prim, KIND_SPHERE, KIND_BOX, KIND_CAPSULE, KIND_TORUS,
    OP_SOLID, OP_SUBTRACT, MAX_PRIMS
)
import time, os
import json
import logging
log = logging.getLogger("adaptivecad.gui")
try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

def load_text(p):
    return Path(p).read_text(encoding="utf-8")

class AnalyticViewport(QOpenGLWidget):
    def __init__(self, parent=None, aacore_scene: AACoreScene | None = None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3,3); fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSamples(4)  # 4x MSAA
        self.setFormat(fmt)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Shared AACore scene (may be injected)
        self.scene = aacore_scene if aacore_scene is not None else AACoreScene()
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
        try:
            self.scene.on_changed(self._on_scene_changed)
        except Exception:
            pass
        if aacore_scene is None and len(self.scene.prims) == 0:
            self._demo_scene()
        # selection
        self.selected_index = -1
        self._picking_fbo = None
        self._pick_tex = None
        self._depth_rb = None
        # micro pick FBO
        self._pick_fbo = None
        # translation drag state
        self._drag_move_active = False
        self._drag_last_pos = None

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
        # create simple FBO for ID picking
        self._init_picking()
        # --- 1x1 FBO for micro picking ---
        try:
            self._pick_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self._pick_fbo)
            self._pick_tex_micro = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._pick_tex_micro)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._pick_tex_micro, 0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        except Exception as e:
            log.debug(f"Micro pick FBO init failed: {e}")

    def _init_picking(self):
        try:
            if self._picking_fbo:
                return
            self._picking_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self._picking_fbo)
            self._pick_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._pick_tex)
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,self.width(),self.height(),0,GL_RGBA,GL_UNSIGNED_BYTE,None)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
            glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,self._pick_tex,0)
            self._depth_rb = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER,self._depth_rb)
            glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,self.width(),self.height())
            glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,self._depth_rb)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                log.debug("Picking FBO incomplete")
            glBindFramebuffer(GL_FRAMEBUFFER,0)
        except Exception as e:
            log.debug(f"Picking FBO init failed: {e}")

    def resizeGL(self, w, h):
        # resize picking attachments
        try:
            if self._pick_tex:
                glBindTexture(GL_TEXTURE_2D, self._pick_tex)
                glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,None)
            if self._depth_rb:
                glBindRenderbuffer(GL_RENDERBUFFER,self._depth_rb)
                glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,w,h)
        except Exception:
            pass

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
        glViewport(0,0,self.width(), self.height())
        glClearColor(*self.scene.bg_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self._render_scene_internal(debug_override=None)

    def _render_scene_internal(self, debug_override=None):
        if not self.prog:
            return
        glUseProgram(self.prog)
        glUniform3f(glGetUniformLocation(self.prog,"u_cam_pos"), *self.cam_pos)
        R = self._cam_basis(); glUniformMatrix3fv(glGetUniformLocation(self.prog,"u_cam_rot"), 1, GL_FALSE, R.T)
        glUniform2f(glGetUniformLocation(self.prog,"u_res"), float(self.width()), float(self.height()))
        glUniform3f(glGetUniformLocation(self.prog,"u_env"), *self.scene.env_light)
        glUniform3f(glGetUniformLocation(self.prog,"u_bg"), *self.scene.bg_color)
        pack = self.scene.to_gpu_structs(max_prims=MAX_PRIMS)
        n = int(pack['count'])
        U = lambda name: glGetUniformLocation(self.prog, name)
        glUniform1i(U("u_count"), n)
        glUniform1iv(U("u_kind"), n, pack['kind'])
        glUniform1iv(U("u_op"), n, pack['op'])
        glUniform1fv(U("u_beta"), n, pack['beta'])
        glUniform3fv(U("u_color"), n, pack['color'])
        glUniform4fv(U("u_params"), n, pack['params'])
        glUniformMatrix4fv(U("u_xform"), n, GL_FALSE, pack['xform'])
        glUniform1i(U("u_mode"), 0)
        mode_val = self.debug_mode if debug_override is None else debug_override
        glUniform1i(U("u_debug"), int(mode_val))
        glUniform1f(U("u_far"), float(self.far_plane))
        glUniform1i(U("u_use_analytic_aa"), int(self.use_analytic_aa))
        glUniform1i(U("u_use_toon"), int(self.use_toon))
        glUniform1f(U("u_toon_levels"), float(self.toon_levels))
        glUniform1f(U("u_curv_strength"), float(self.curv_strength))
        glUniform1i(U("u_use_foveated"), int(self.use_foveated))
        glUniform1i(U("u_show_beta_overlay"), int(self.show_beta_overlay))
        glUniform1f(U("u_beta_overlay_intensity"), float(self.beta_overlay_intensity))
        glUniform1f(U("u_beta_scale"), float(self.beta_scale))
        glUniform1i(U("u_beta_cmap"), int(self.beta_cmap))
        glUniform1i(U("u_selected"), int(self.selected_index))
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def _draw_frame(self, debug_override=None):  # retained for other callers
        glViewport(0,0,self.width(), self.height())
        glClearColor(*self.scene.bg_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self._render_scene_internal(debug_override)

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
            log.info(self._shortcuts_help())
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
            # --- Keyboard nudges (view-aligned + world-aligned) ---
            pr = self._current_prim()
            if pr is not None:
                # determine step
                step_widget = getattr(self.parent(), '_nudge_step', None)
                try:
                    step = float(step_widget.value()) if step_widget is not None else 1.0
                except Exception:
                    step = 1.0
                moved = False
                basis = self._cam_basis()
                pos_vec = pr.xform.M[:3,3].astype(np.float32)
                # View-aligned (WASD + Q/E)
                if k == Qt.Key_W: pos_vec += basis[:,1]*step; moved=True
                elif k == Qt.Key_S: pos_vec -= basis[:,1]*step; moved=True
                elif k == Qt.Key_D: pos_vec += basis[:,0]*step; moved=True
                elif k == Qt.Key_A: pos_vec -= basis[:,0]*step; moved=True
                elif k == Qt.Key_E: pos_vec -= basis[:,2]*step; moved=True  # forward (-Z in basis third column sign depending)
                elif k == Qt.Key_Q: pos_vec += basis[:,2]*step; moved=True
                # World-aligned (arrows + PgUp/PgDn)
                elif k == Qt.Key_Up: pos_vec += np.array([0,step,0], np.float32); moved=True
                elif k == Qt.Key_Down: pos_vec -= np.array([0,step,0], np.float32); moved=True
                elif k == Qt.Key_Right: pos_vec += np.array([step,0,0], np.float32); moved=True
                elif k == Qt.Key_Left: pos_vec -= np.array([step,0,0], np.float32); moved=True
                elif k == Qt.Key_PageUp: pos_vec += np.array([0,0,step], np.float32); moved=True
                elif k == Qt.Key_PageDown: pos_vec -= np.array([0,0,step], np.float32); moved=True
                if moved:
                    self._apply_panel_move(pos_vec)
                    return
            super().keyPressEvent(event)

    # --- Mouse Interaction ---
    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse = event.position()
        self._last_buttons = event.buttons()
        # Plain LMB (no modifiers) -> pick
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            # Use full-resolution FBO picking for accurate coordinate-based selection.
            try:
                self._perform_pick(int(event.position().x()), int(event.position().y()))
                pid = self.selected_index
                if pid >= 0:
                    try:
                        if hasattr(self.parent(), '_select_prim'):
                            self.parent()._select_prim(pid)
                    except Exception:
                        pass
                self.update()
            except Exception as e:
                log.debug(f"mousePressEvent pick failed: {e}")
        # Shift + LMB -> begin move-drag if a primitive selected
        if (event.button() == Qt.MouseButton.LeftButton and \
            (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and \
            self._current_prim() is not None):
            self._drag_move_active = True
            self._drag_last_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None:
            self._last_mouse = event.position()
        dx = event.position().x() - self._last_mouse.x()
        dy = event.position().y() - self._last_mouse.y()
        # Screen-plane translate when active
        if self._drag_move_active and self._drag_last_pos is not None:
            dx2 = event.position().x() - self._drag_last_pos.x()
            dy2 = event.position().y() - self._drag_last_pos.y()
            R = self._cam_basis(); right = R[:,0]; up = R[:,1]
            step = self.distance * 0.0015
            dp = right * dx2 * step - up * dy2 * step
            pr = self._current_prim()
            if pr is not None:
                new_pos = pr.xform.M[:3,3] + dp.astype(np.float32)
                self._apply_panel_move(new_pos)
            self._drag_last_pos = event.position()
            return
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

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_move_active = False
            self._drag_last_pos = None
        super().mouseReleaseEvent(event)

    # --- Selection + panel sync helpers ---
    def _current_prim(self):
        try:
            panel = self.parent()
            if hasattr(panel, '_current_sel') and 0 <= panel._current_sel < len(self.scene.prims):
                return self.scene.prims[panel._current_sel]
        except Exception:
            return None
        return None

    def _apply_panel_move(self, new_pos):
        panel = self.parent()
        try:
            if hasattr(panel, '_sp_move'):
                for i,v in enumerate(new_pos):
                    panel._sp_move[i].blockSignals(True)
                    panel._sp_move[i].setValue(float(v))
                    panel._sp_move[i].blockSignals(False)
                if hasattr(panel, '_apply_edit'):
                    panel._apply_edit()
        except Exception:
            pass

    def _perform_pick(self, x:int, y:int):
        if not self.prog or self._picking_fbo is None:
            return
        try:
            # Ensure GL context is current for FBO operations outside paintGL
            try:
                self.makeCurrent()
            except Exception:
                pass
            dpr = self.devicePixelRatio() if hasattr(self, 'devicePixelRatio') else 1
            sx = int(x * dpr); sy = int(y * dpr)
            glBindFramebuffer(GL_FRAMEBUFFER, self._picking_fbo)
            glViewport(0,0,self.width(), self.height())
            glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.prog)
            # draw IDs
            glUniform3f(glGetUniformLocation(self.prog,"u_cam_pos"), *self.cam_pos)
            R = self._cam_basis(); glUniformMatrix3fv(glGetUniformLocation(self.prog,"u_cam_rot"), 1, GL_FALSE, R.T)
            glUniform2f(glGetUniformLocation(self.prog,"u_res"), float(self.width()), float(self.height()))
            glUniform3f(glGetUniformLocation(self.prog,"u_env"), *self.scene.env_light)
            glUniform3f(glGetUniformLocation(self.prog,"u_bg"), 0,0,0)
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
            glUniform1i(glGetUniformLocation(self.prog, "u_mode"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_debug"), 2)  # ID mode
            glUniform1i(glGetUniformLocation(self.prog, "u_use_analytic_aa"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_use_toon"), 0)
            glUniform1f(glGetUniformLocation(self.prog, "u_toon_levels"), 1.0)
            glUniform1f(glGetUniformLocation(self.prog, "u_curv_strength"), 0.0)
            glUniform1i(glGetUniformLocation(self.prog, "u_use_foveated"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_show_beta_overlay"), 0)
            glUniform1f(glGetUniformLocation(self.prog, "u_beta_overlay_intensity"), 0.0)
            glUniform1f(glGetUniformLocation(self.prog, "u_beta_scale"), 1.0)
            glUniform1i(glGetUniformLocation(self.prog, "u_beta_cmap"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_selected"), -1)
            glBindVertexArray(self._vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            px = np.zeros((1,1,4), dtype=np.uint8)
            ry = self.height() - 1 - (sy if dpr!=1 else y)
            glReadPixels(sx if dpr!=1 else x, ry, 1,1, GL_RGBA, GL_UNSIGNED_BYTE, px)
            enc = int(px[0,0,0]) | (int(px[0,0,1])<<8) | (int(px[0,0,2])<<16)
            if enc==0:
                self.selected_index = -1
            else:
                self.selected_index = enc-1
            glBindFramebuffer(GL_FRAMEBUFFER,0)
            self.update()
            log.debug(f"pick raw={px[0,0].tolist()} enc={enc} sel={self.selected_index} at ({x},{y}) dpr={dpr}")
        except Exception as e:
            log.debug(f"pick failed (full-res) falling back to micro pick: {e}")
            try:
                pid = self._pick_id_at(x,y)
                self.selected_index = pid
            except Exception:
                pass

    def _pick_id_at(self, x:int, y:int) -> int:
        if not self.prog or self._pick_fbo is None:
            return -1
        try:
            # simplistic: render full scene in ID mode into 1x1 buffer (center ray)
            glBindFramebuffer(GL_FRAMEBUFFER, self._pick_fbo)
            glViewport(0,0,1,1)
            glClearColor(0,0,0,1)
            glClear(GL_COLOR_BUFFER_BIT)
            self._render_scene_internal(debug_override=2)
            data = glReadPixels(0,0,1,1,GL_RGBA,GL_UNSIGNED_BYTE)
            import numpy as _np
            px = _np.frombuffer(data, dtype=_np.uint8)
            enc = int(px[0])
            pid = enc-1
            glBindFramebuffer(GL_FRAMEBUFFER,0)
            return pid if 0 <= pid < len(self.scene.prims) else -1
        except Exception as e:
            log.debug(f"micro pick failed: {e}")
            try: glBindFramebuffer(GL_FRAMEBUFFER,0)
            except Exception: pass
            return -1

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
            "Click : select primitive (highlight)\n"
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
            log.info(f"Saved {label} -> {path}")
        self.debug_mode = prev_mode
        self.update()


class AnalyticViewportPanel(QWidget):
    """Composite widget: analytic SDF viewport + control panel."""
    def __init__(self, parent=None, aacore_scene: AACoreScene | None = None):
        super().__init__(parent)
        self.view = AnalyticViewport(self, aacore_scene=aacore_scene)
        self._build_ui()
        self.setWindowTitle("Analytic Viewport (Panel)")
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4,4,4,4)
        root.addWidget(self.view, 1)
        side = QVBoxLayout(); side.setSpacing(6)

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
        btn_help = QPushButton("Print Shortcuts [H]"); btn_help.clicked.connect(lambda: log.info(self.view._shortcuts_help()))
        va.addWidget(btn_save); va.addWidget(btn_help); side.addWidget(gb_act)

        # --- Environment ---
        gb_env = QGroupBox("Environment"); fe = QFormLayout(); gb_env.setLayout(fe)
        self._bg_sliders = []
        for i, ch in enumerate(['R','G','B']):
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(0,255); s.setValue(int(self.view.scene.bg_color[i]*255)); s.valueChanged.connect(self._on_bg_changed)
            fe.addRow(f"BG {ch}", s); self._bg_sliders.append(s)
        self._env_sliders = []
        for i, ch in enumerate(['X','Y','Z']):
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(-150,150); s.setValue(int(self.view.scene.env_light[i]*100)); s.valueChanged.connect(self._on_env_changed)
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
        from PySide6.QtWidgets import QDoubleSpinBox, QComboBox as _QCB2, QColorDialog
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
        # Rotation & Scale controls
        def rspin():
            s = QDoubleSpinBox(); s.setRange(-180.0,180.0); s.setSingleStep(1.0); s.setDecimals(2); s.setValue(0.0); return s
        def sspin():
            s = QDoubleSpinBox(); s.setRange(0.01,100.0); s.setSingleStep(0.05); s.setDecimals(3); s.setValue(1.0); return s
        self._sp_rot = [rspin() for _ in range(3)]
        self._sp_scl = [sspin() for _ in range(3)]
        # Move (translate) controls
        def mspin():
            s = QDoubleSpinBox(); s.setRange(-1000.0,1000.0); s.setSingleStep(0.01); s.setDecimals(4); s.setValue(0.0); return s
        self._sp_move = [mspin() for _ in range(3)]
        fe2.addRow("Rot ° X/Y/Z", self._make_row(self._sp_rot))
        fe2.addRow("Scale X/Y/Z", self._make_row(self._sp_scl))
        fe2.addRow("Move X/Y/Z", self._make_row(self._sp_move))
        from PySide6.QtWidgets import QDoubleSpinBox as _QDB2
        self._nudge_step = _QDB2(); self._nudge_step.setRange(0.001,100.0); self._nudge_step.setDecimals(3); self._nudge_step.setSingleStep(0.1); self._nudge_step.setValue(1.0)
        fe2.addRow("Nudge Step", self._nudge_step)
        self._edit_group.setEnabled(False); vp.addWidget(self._edit_group)
        for sp in self._sp_pos: sp.valueChanged.connect(self._apply_edit)
        for sp in self._sp_param: sp.valueChanged.connect(self._apply_edit)
        self._sp_beta.valueChanged.connect(self._apply_edit); self._op_box.currentIndexChanged.connect(self._apply_edit)
        for s in self._sp_rot + self._sp_scl + self._sp_move:
            s.valueChanged.connect(self._apply_edit)
        def pick_color():
            from PySide6.QtGui import QColor
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
        # populate move spinners (authoritative for translation edits)
        if hasattr(self, '_sp_move'):
            for i in range(3):
                self._sp_move[i].blockSignals(True)
                self._sp_move[i].setValue(float(pos[i]))
                self._sp_move[i].blockSignals(False)
        # params
        self._sp_param[0].blockSignals(True); self._sp_param[0].setValue(float(pr.params[0])); self._sp_param[0].blockSignals(False)
        self._sp_param[1].blockSignals(True); self._sp_param[1].setValue(float(pr.params[1] if pr.kind!=KIND_SPHERE else 0.0)); self._sp_param[1].blockSignals(False)
        self._sp_beta.blockSignals(True); self._sp_beta.setValue(pr.beta); self._sp_beta.blockSignals(False)
        self._op_box.blockSignals(True); self._op_box.setCurrentIndex(0 if pr.op=='solid' else 1); self._op_box.blockSignals(False)
        self._current_color = tuple(pr.color[:3])
        # rotation / scale sync
        if hasattr(pr, 'euler'):
            for i in range(3):
                self._sp_rot[i].blockSignals(True); self._sp_rot[i].setValue(float(pr.euler[i])); self._sp_rot[i].blockSignals(False)
        if hasattr(pr, 'scale'):
            for i in range(3):
                self._sp_scl[i].blockSignals(True); self._sp_scl[i].setValue(float(pr.scale[i])); self._sp_scl[i].blockSignals(False)
        self._edit_group.setEnabled(True)

    def _apply_edit(self):
        if self._current_sel < 0 or self._current_sel >= len(self.view.scene.prims):
            return
        pr = self.view.scene.prims[self._current_sel]
        # existing matrix before rebuilding
        M = pr.xform.M.copy()
        # apply params
        pr.params[0] = self._sp_param[0].value()
        if pr.kind != KIND_SPHERE:
            pr.params[1] = self._sp_param[1].value()
        pr.beta = self._sp_beta.value()
        pr.op = 'solid' if self._op_box.currentIndex()==0 else 'subtract'
        pr.color[:3] = np.array(self._current_color[:3])
        # translation driven by move spinners
        if hasattr(self, '_sp_move'):
            pos = [self._sp_move[i].value() for i in range(3)]
        else:
            pos = [self._sp_pos[i].value() for i in range(3)]
        rx, ry, rz = [self._sp_rot[i].value() for i in range(3)]
        sx, sy, sz = [self._sp_scl[i].value() for i in range(3)]
        if hasattr(pr, 'set_transform'):
            pr.set_transform(pos=pos, euler=[rx,ry,rz], scale=[sx,sy,sz])
        else:
            # fallback: just translate
            for i in range(3): M[i,3] = pos[i]
            pr.xform.M = M
        # notify & refresh
        self.view.scene._notify()
        self.view.update()

    # Convenience passthroughs
    def viewport(self) -> AnalyticViewport: return self.view

def create_analytic_viewport_with_panel(parent=None, aacore_scene: AACoreScene | None = None):
    return AnalyticViewportPanel(parent, aacore_scene=aacore_scene)

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
        data_update={
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
        # Preserve unrelated preference keys (e.g., 'analytic_as_main') by merging
        existing = {}
        if os.path.exists(self._settings_path):
            try:
                with open(self._settings_path,'r',encoding='utf-8') as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(data_update)
        with open(self._settings_path,'w',encoding='utf-8') as f:
            json.dump(existing,f,indent=2)
    except Exception as e:
        print(f"(analytic settings) save failed: {e}")

# Monkey-patch methods onto AnalyticViewport class after definition (simple, avoids editing earlier class body heavily)
AnalyticViewport._load_settings = _load_settings
AnalyticViewport._save_settings = _save_settings
