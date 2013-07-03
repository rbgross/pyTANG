import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

class Light:
    def __init__(self, renderer):
        self.renderer = renderer
        self.lightPos = np.array([0.0, -1.0, -1.0, 0.0], dtype = np.float32)
        self.renderer.setLightPos(self.lightPos)
