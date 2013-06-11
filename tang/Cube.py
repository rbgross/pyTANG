import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

from Mesh import Mesh

class Cube:
    def __init__(self, renderer):
        self.renderer = renderer
        self.mesh = Mesh(os.path.abspath(os.path.join(self.renderer.resPath, 'models', 'CubeTest2.obj')))  # 'C:\\Users\\Ryan\\Game Tests\\CubeTest2.obj'
        self.position = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        self.color = np.array([0.0, 0.0, 0.0], dtype = np.float32)

    def draw(self):
        model = hm.translation(self.renderer.environment.model, self.position)
        self.renderer.setModel(model)
        self.renderer.setDiffCol(self.color)
        self.mesh.draw()
