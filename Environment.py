import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np

class Environment:
    def __init__(self, renderer):
        self.renderer = renderer
        self.model = glm.mat4()
        self.light = Light(self.renderer)
        self.focusObject = FocusObject(self.renderer)
        self.focusObject.color = glm.vec3(1.0, 1.0, 1.0)

        self.cubes = []
        for i in xrange(0, 8):
            self.cubes.append(Cube(self.renderer))

        self.cubes[0].position = glm.vec3( 10.0,  10.0,  10.0)
        self.cubes[1].position = glm.vec3( 10.0,  10.0, -10.0)
        self.cubes[2].position = glm.vec3( 10.0, -10.0,  10.0)
        self.cubes[3].position = glm.vec3( 10.0, -10.0, -10.0)
        self.cubes[4].position = glm.vec3(-10.0,  10.0,  10.0)
        self.cubes[5].position = glm.vec3(-10.0,  10.0, -10.0)
        self.cubes[6].position = glm.vec3(-10.0, -10.0,  10.0)
        self.cubes[7].position = glm.vec3(-10.0, -10.0, -10.0)

        self.cubes[0].color = glm.vec3(0.3, 0.3, 1.0)
        self.cubes[1].color = glm.vec3(1.0, 0.3, 0.3)
        self.cubes[2].color = glm.vec3(0.3, 1.0, 0.3)
        self.cubes[3].color = glm.vec3(1.0, 1.0, 0.3)
        self.cubes[4].color = glm.vec3(1.0, 1.0, 0.3)
        self.cubes[5].color = glm.vec3(0.3, 1.0, 0.3)
        self.cubes[6].color = glm.vec3(1.0, 0.3, 0.3)
        self.cubes[7].color = glm.vec3(0.3, 0.3, 1.0)

        self.datapoints = []

    def addDataPoint(self, position):
        dataPoint = DataPoint(self.renderer)
        dataPoint.position = position
        dataPoint.color = glm.vec3(1.0, 1.0, 1.0)
        self.dataPoints.append(dataPoint)

    def draw(self):
        if not self.renderer.hideCube:
            for i in xrange(0, 8):
                self.cubes[i].draw()

        self.focusObject.draw()
        for i in xrange(0, len(self.dataPoints)):
            self.dataPoints[i].draw()
