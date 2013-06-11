import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np

class Light:
    def __init__(self, renderer):
        self.renderer = renderer
        self.lightPos = glm.vec4(0.0, 1.0, 1.0, 0.0)
        self.renderer.setLightPos(self.lightPos)
