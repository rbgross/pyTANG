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
        self.readData(os.path.abspath(os.path.join(self.renderer.resPath, 'data', 'DragonScene.txt')))

    def readData(self, fileName):
        self.actors = []
        f = open(fileName, 'r')
        for line in f: 
            s = line.split()
            if len(s) > 0:
                if s[0] == 'Cube':
                    temp = self.actorFactory.makeCube()
                if s[0] == 'Edge':
                    temp = self.actorFactory.makeEdge()                
                if s[0] == 'DataPoint':
                    temp = self.actorFactory.makeDataPoint()
                elif s[0] == 'Dragon':
                    temp = self.actorFactory.makeDragon()
                temp.position = np.array([s[2], s[3], s[4]], dtype = np.float32)
                temp.rotation = np.array([s[6], s[7], s[8]], dtype = np.float32)
                temp.color = np.array([s[10], s[11], s[12]], dtype = np.float32)
                self.actors.append(temp)

    def draw(self):
        if not self.hideCube:
            for i in xrange(0, 8):
                self.actors[i].draw()

        for i in xrange(8, len(self.actors)):
            self.actors[i].draw()
