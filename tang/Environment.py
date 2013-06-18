import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

from Light import Light
from ActorFactory import ActorFactory

class Environment:
    def __init__(self, renderer):
        self.hideCube = False
        self.renderer = renderer
        self.model = hm.identity()
        self.light = Light(self.renderer)
        
        self.actorFactory = ActorFactory(self.renderer, self)

        self.cubes = []
        self.cubes.append(self.actorFactory.makeBlueCube())
        self.cubes.append(self.actorFactory.makeRedCube())
        self.cubes.append(self.actorFactory.makeGreenCube())
        self.cubes.append(self.actorFactory.makeYellowCube())
        self.cubes.append(self.actorFactory.makeYellowCube())
        self.cubes.append(self.actorFactory.makeGreenCube())
        self.cubes.append(self.actorFactory.makeRedCube())
        self.cubes.append(self.actorFactory.makeBlueCube())

        self.cubes[0].position = np.array([10.0, 10.0, 10.0], dtype = np.float32)
        self.cubes[1].position = np.array([10.0, 10.0, -10.0], dtype = np.float32)
        self.cubes[2].position = np.array([10.0, -10.0, 10.0], dtype = np.float32)
        self.cubes[3].position = np.array([10.0, -10.0, -10.0], dtype = np.float32)
        self.cubes[4].position = np.array([-10.0, 10.0, 10.0], dtype = np.float32)
        self.cubes[5].position = np.array([-10.0, 10.0, -10.0], dtype = np.float32)
        self.cubes[6].position = np.array([-10.0, -10.0, 10.0], dtype = np.float32)
        self.cubes[7].position = np.array([-10.0, -10.0, -10.0], dtype = np.float32)

        self.readData(os.path.abspath(os.path.join(self.renderer.resPath, 'data', 'PointData.txt')))

    def readData(self, fileName):
        self.dataPoints = []
        f = open(fileName, 'r')
        for line in f: 
            s = line.split()
            dataPoint = self.actorFactory.makeDataPoint()
            dataPoint.position = np.array([s[0], s[1], s[2]], dtype = np.float32)
            self.dataPoints.append(dataPoint)

    def draw(self):
        if not self.hideCube:
            for i in xrange(0, 8):
                self.cubes[i].draw()

        for i in xrange(0, len(self.dataPoints)):
            self.dataPoints[i].draw()
