import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np

class DataPoint:
    def __init__(self, renderer):
        self.renderer = renderer
        self.mesh = Mesh('C:\\Users\\Ryan\\Game Tests\\SphereTest.obj')
        self.position = glm.vec3()
        self.color = glm.vec3()

    def draw(self):
        model = glm.translate(self.renderer.environment.model, self.position)
        model = glm.scale(model, glm.vec3(0.05, 0.05, 0.05))
        self.renderer.setModel(model)
        self.renderer.setDiffCol(self.color)
        self.mesh.draw()
