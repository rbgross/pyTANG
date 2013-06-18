import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import math
import numpy as np
import hommat as hm

class Input:
    def __init__(self, environment):
        self.environment = environment
        
        self.wheelPosition = glfw.GetMouseWheel()
        self.oldMouseX, self.oldMouseY = glfw.GetMousePos()
        self.curMouseX = self.oldMouseX
        self.curMouseY = self.oldMouseY
        self.hideCube = False
        self.leftPressed = False
        self.rightPressed = False

    def pollInput(self):
        tempWheelPosition = glfw.GetMouseWheel()
        if tempWheelPosition != self.wheelPosition:
            self.wheelPosition = tempWheelPosition
            self.setView(hm.lookat(hm.identity(), np.array([0.0, 0.0, 55.0 - self.wheelPosition, 1.0], dtype = np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32)))

        if glfw.GetKey('A'):
            self.environment.model = hm.rotation(self.environment.model, -1, [0, 1, 0])

        if glfw.GetKey('D'):
            self.environment.model = hm.rotation(self.environment.model, 1, [0, 1, 0])

        if glfw.GetKey('W'):
            self.environment.model = hm.rotation(self.environment.model, -1, [1, 0, 0])

        if glfw.GetKey('S'):
            self.environment.model = hm.rotation(self.environment.model, 1, [1, 0, 0])

        if glfw.GetKey('Q'):
            self.environment.model = hm.rotation(self.environment.model, -1, [0, 0, 1])

        if glfw.GetKey('E'):
            self.environment.model = hm.rotation(self.environment.model, 1, [0, 0, 1])
        
        if not self.leftPressed and glfw.GetMouseButton(glfw.MOUSE_BUTTON_LEFT):
            self.leftPressed = True
            self.hideCube = not self.hideCube

        if not glfw.GetMouseButton(glfw.MOUSE_BUTTON_LEFT):
            self.leftPressed = False

        #if not self.rightPressed and glfw.GetMouseButton(glfw.MOUSE_BUTTON_RIGHT):
            #self.rightPressed = True
            #self.oldMouseX, self.oldMouseY = glfw.GetMousePos()
            #self.curMouseX = self.oldMouseX
            #self.curMouseY = self.oldMouseY

        #if not glfw.GetMouseButton(glfw.MOUSE_BUTTON_RIGHT):
            #self.rightPressed = False

        #if self.rightPressed:
            #self.curMouseX, self.curMouseY = glfw.GetMousePos()
            #if self.curMouseX != self.oldMouseX or self.curMouseY != self.oldMouseY:
                #oldVec = self.calcArcBallVector(self.oldMouseX, self.oldMouseY)
                #curVec = self.calcArcBallVector(self.curMouseX, self.curMouseY)
                #angle = math.acos(min(1.0, np.dot(oldVec, curVec)))
                #cameraAxis = np.cross(oldVec, curVec)
                #cameraToObjectCoords = np.linalg.inv(np.dot(self.view[:-1,:-1], self.environment.model[:-1,:-1]))
                #cameraAxisObjectCoords = np.dot(cameraToObjectCoords, cameraAxis)
                #self.environment.model = hm.rotation(self.environment.model, math.degrees(angle), cameraAxisObjectCoords)
                #self.oldMouseX = self.curMouseX
                #self.oldMouseY = self.curMouseY

    #def calcArcBallVector(self, mouseX, mouseY):
        #vec = np.array([float(mouseX) / self.windowWidth * 2 - 1.0, float(mouseY) / self.windowHeight * 2 - 1.0, 0], dtype = np.float32)
        #vec[1] = -vec[1]
        #distSquared = vec[0] * vec[0] + vec[1] * vec[1]
        #if distSquared <= 1.0:
            #vec[2] = math.sqrt(1.0 - distSquared)
        #else:
            #vec *= 1 / np.linalg.norm(vec)
        #return vec
