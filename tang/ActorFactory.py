import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

from Actor import Actor
from Mesh import Mesh

class ActorFactory:
    def __init__(self, renderer, environment):
        self.renderer = renderer
        self.environment = environment
        self.loadResources()        

    def loadResources(self):
        self.cubeMesh = Mesh(os.path.abspath(os.path.join(self.renderer.resPath, 'models', 'Cube.obj')))
        self.sphereMesh = Mesh(os.path.abspath(os.path.join(self.renderer.resPath, 'models', 'SmallSphere.obj')))

    def makeCube(self):
        cube = Actor(self.renderer, self.environment, self.cubeMesh)
        return cube

    def makeDataPoint(self):
        dataPoint = Actor(self.renderer, self.environment, self.sphereMesh)
        return dataPoint
