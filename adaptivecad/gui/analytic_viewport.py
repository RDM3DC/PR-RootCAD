# adaptivecad/gui/analytic_viewport.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent
from PyQt6.QtCore import Qt, QSize
from OpenGL.GL import *
import numpy as np
from pathlib import Path
from adaptivecad.analytic.scene import Scene, Primitive, Transform

def load_text(p):
    return Path(p).read_text(encoding="utf-8")

class AnalyticViewport(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3,3); fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.setFormat(fmt)
        self.scene = Scene()
        # camera
        self.cam_pos = np.array([0.0, 0.0, 6.0], np.float32)
        self.yaw=0.0; self.pitch=0.0
        self.last = None
        # shader handles
        self.prog = None
        # build a tiny demo scene
        self._demo_scene()

    def _demo_scene(self):
        # sphere
        self.scene.primitives.append(Primitive(
            kind='sphere', params=np.array([0,0,0,1.2], np.float32),
            color=np.array([0.9,0.4,0.3], np.float32), pia_beta=0.12, object_id=10
        ))
        # capsule
        self.scene.primitives.append(Primitive(
            kind='capsule', params=np.array([1.0,0,0,0.4], np.float32),
            color=np.array([0.3,0.7,0.9], np.float32), pia_beta=0.0, object_id=11,
            xform=Transform(np.array([[1,0,0, -1.8],
                                      [0,1,0,  0.0],
                                      [0,0,1,  0.0],
                                      [0,0,0,  1.0]], np.float32))
        ))
        # torus
        self.scene.primitives.append(Primitive(
            kind='torus', params=np.array([1.2,0.25,0,0], np.float32),
            color=np.array([0.6,0.9,0.5], np.float32), pia_beta=0.05, object_id=12,
            xform=Transform(np.array([[1,0,0, 1.8],
                                      [0,1,0, 0.0],
                                      [0,0,1, 0.0],
                                      [0,0,0, 1.0]], np.float32))
        ))

    def sizeHint(self): return QSize(800,600)

    # --- OpenGL lifecycle ---
    def initializeGL(self):
        glDisable(GL_DEPTH_TEST)
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

    def paintGL(self):
        glViewport(0,0,self.width(), self.height())
        glClearColor(*self.scene.bg_color, 1.0); glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.prog)
        # uniforms
        glUniform3f(glGetUniformLocation(self.prog,"u_cam_pos"), *self.cam_pos)
        R = self._cam_basis(); glUniformMatrix3fv(glGetUniformLocation(self.prog,"u_cam_rot"), 1, GL_FALSE, R.T)
        glUniform2f(glGetUniformLocation(self.prog,"u_res"), float(self.width()), float(self.height()))
        glUniform1f(glGetUniformLocation(self.prog,"u_env"), float(self.scene.env_light))
        glUniform3f(glGetUniformLocation(self.prog,"u_bg"), *self.scene.bg_color)
        # upload scene
        gpu = self.scene.to_gpu_structs()
        glUniform1i(glGetUniformLocation(self.prog,"u_count"), int(gpu['n']))
        def U(name): return glGetUniformLocation(self.prog, name)
        glUniform1iv(U("u_kind"), len(gpu['kinds']), gpu['kinds'])
        glUniform4fv(U("u_params"), gpu['params'].size//4, gpu['params'])
        glUniform3fv(U("u_color"),  gpu['colors'].size//3, gpu['colors'])
        glUniform1fv(U("u_beta"),   len(gpu['betas']), gpu['betas'])
        glUniform1iv(U("u_id"),     len(gpu['ids']), gpu['ids'])
        glUniformMatrix4fv(U("u_xform"), gpu['xforms'].size//16, GL_FALSE, gpu['xforms'])
        # draw (beauty)
        glUniform1i(glGetUniformLocation(self.prog,"u_mode"), 0)
        glBindVertexArray(self._vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    # --- input ---
    def mousePressEvent(self, e:QMouseEvent):
        self.last = (e.position().x(), e.position().y())
    def mouseMoveEvent(self, e:QMouseEvent):
        if self.last is None: return
        x,y = e.position().x(), e.position().y()
        dx, dy = (x-self.last[0]), (y-self.last[1])
        if e.buttons() & Qt.LeftButton:
            self.yaw   += dx * 0.005
            self.pitch += dy * 0.005
            self.pitch = np.clip(self.pitch, -1.2, 1.2)
        elif e.buttons() & Qt.RightButton:
            right = self._cam_basis()[:,0]; up = self._cam_basis()[:,1]
            self.cam_pos += (-right*dx + up*dy) * 0.01
        self.last = (x,y); self.update()
    def wheelEvent(self, e:QWheelEvent):
        self.cam_pos += self._cam_basis()[:,2]*(-e.angleDelta().y()/120.0)*0.25
        self.update()
