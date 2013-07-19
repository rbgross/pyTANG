import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

class Actor:
    def __init__(self, renderer, environment, mesh):
        self.renderer = renderer
        self.environment = environment

        #Mesh component
        self.mesh = mesh

        #Transform component
        self.position = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        self.scale = np.array([1.0, 1.0, 1.0], dtype = np.float32)

        #Material component
        self.color = np.array([0.0, 0.0, 0.0], dtype = np.float32)

    def draw(self):
        model = hm.translation(self.environment.model, self.position)
        model = hm.rotation(model, self.rotation[0], [1, 0, 0])
        model = hm.rotation(model, self.rotation[1], [0, 1, 0])
        model = hm.rotation(model, self.rotation[2], [0, 0, 1])
        self.renderer.setModel(model)
        self.renderer.setDiffCol(self.color)
        self.mesh.draw()
