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
        self.sphereMesh = Mesh(os.path.abspath(os.path.join(self.renderer.resPath, 'models', 'Sphere.obj')))

    def makeBlueCube(self):
        cube = Actor(self.renderer, self.environment, self.cubeMesh)
        cube.color = np.array([0.3, 0.3, 1.0], dtype = np.float32)
        return cube

    def makeRedCube(self):
        cube = Actor(self.renderer, self.environment, self.cubeMesh)
        cube.color = np.array([1.0, 0.3, 0.3], dtype = np.float32)
        return cube

    def makeGreenCube(self):
        cube = Actor(self.renderer, self.environment, self.cubeMesh)
        cube.color = np.array([0.3, 1.0, 0.3], dtype = np.float32)
        return cube

    def makeYellowCube(self):
        cube = Actor(self.renderer, self.environment, self.cubeMesh)
        cube.color = np.array([1.0, 1.0, 0.3], dtype = np.float32)
        return cube

    def makeDataPoint(self):
        dataPoint = Actor(self.renderer, self.environment, self.sphereMesh)
        dataPoint.color = np.array([1.0, 1.0, 1.0], dtype = np.float32)
        return dataPoint
