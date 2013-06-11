import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np

class FocusObject:
    def __init__(self, renderer):
        self.renderer = renderer
        self.mesh = Mesh('C:\\Users\\Ryan\\Game Tests\\Dragon.obj')
        self.position = glm.vec3()
        self.color = glm.vec3()

    def draw(self):
        model = glm.translate(self.renderer.environment.model, self.position)
        self.renderer.setModel(model)
        self.renderer.setDiffCol(self.color)
        self.mesh.draw()
