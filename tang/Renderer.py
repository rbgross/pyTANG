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

from Environment import Environment
from ColorShader import ColorShader

class Renderer:
    def __init__(self, resPath=None):
        self.resPath = resPath if resPath is not None else os.getcwd()  # default to current directory if None passed
        
        self.windowWidth = 640
        self.windowHeight = 480

        self.initialize()

        self.colorShader = ColorShader()

        self.view = hm.lookat(hm.identity(), np.array([0.0, 0.0, 55.0, 1.0], dtype = np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32))
        self.setView(self.view)

        self.proj = hm.perspective(hm.identity(), 45, float(self.windowWidth) / self.windowHeight, 0.1, 1000.0)
        self.setProj(self.proj)

        self.environment = Environment(self)

        self.wheelPosition = glfw.GetMouseWheel()
        self.oldMouseX, self.oldMouseY = glfw.GetMousePos()
        self.curMouseX = self.oldMouseX
        self.curMouseY = self.oldMouseY
        self.hideCube = False
        self.leftPressed = False
        self.rightPressed = False

        self.readData(os.path.abspath(os.path.join(self.resPath, 'data', 'PointData.txt')))  # 'C:\\Users\\Ryan\\Game Tests\\Data.txt'

    def setModel(self, model):
        self.colorShader.setModel(model)

    def setView(self, view):
        self.colorShader.setView(view)

    def setProj(self, proj):
        self.colorShader.setProj(proj)

    def setLightPos(self, lightPos):
        self.colorShader.setLightPos(lightPos)

    def setDiffCol(self, diffCol):
        self.colorShader.setDiffCol(diffCol)

    def windowOpen(self):
        return glfw.GetWindowParam(glfw.OPENED)

    def draw(self):
        self.pollInput()
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.environment.draw()

        glfw.SwapBuffers()

    def initialize(self):
        glfw.Init()
        glfw.OpenWindowHint(glfw.FSAA_SAMPLES, 4)
        glfw.OpenWindowHint(glfw.OPENGL_VERSION_MAJOR, 3)
        glfw.OpenWindowHint(glfw.OPENGL_VERSION_MINOR, 2)
        glfw.OpenWindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.OpenWindowHint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.OpenWindowHint(glfw.WINDOW_NO_RESIZE, GL_TRUE)
        glfw.OpenWindow(self.windowWidth, self.windowHeight, 0, 0, 0, 0, 0, 0, glfw.WINDOW)
        glfw.SetWindowTitle("TANG")

        glEnable(GL_DEPTH_TEST)
	glEnable(GL_CULL_FACE)

    def pollInput(self):
        tempWheelPosition = glfw.GetMouseWheel()
        if tempWheelPosition != self.wheelPosition:
            self.wheelPosition = tempWheelPosition
            self.setView(hm.lookat(hm.identity(), np.array([0.0, 0.0, 55.0 - self.wheelPosition, 1.0], dtype = np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32)))

        if not self.leftPressed and glfw.GetMouseButton(glfw.MOUSE_BUTTON_LEFT):
            self.leftPressed = True
            self.hideCube = not self.hideCube

        if not glfw.GetMouseButton(glfw.MOUSE_BUTTON_LEFT):
            self.leftPressed = False

        if not self.rightPressed and glfw.GetMouseButton(glfw.MOUSE_BUTTON_RIGHT):
            self.rightPressed = True
            self.oldMouseX, self.oldMouseY = glfw.GetMousePos()
            self.curMouseX = self.oldMouseX
            self.curMouseY = self.oldMouseY

        if not glfw.GetMouseButton(glfw.MOUSE_BUTTON_RIGHT):
            self.rightPressed = False

        if self.rightPressed:
            print self.environment.model
            self.curMouseX, self.curMouseY = glfw.GetMousePos()
            if self.curMouseX != self.oldMouseX or self.curMouseY != self.oldMouseY:
                oldVec = self.calcArcBallVector(self.oldMouseX, self.oldMouseY)
                curVec = self.calcArcBallVector(self.curMouseX, self.curMouseY)
                angle = math.acos(min(1.0, np.dot(oldVec, curVec)))
                cameraAxis = np.cross(oldVec, curVec)
                cameraToObjectCoords = np.linalg.inv(np.dot(self.view[:-1,:-1], self.environment.model[:-1,:-1]))
                cameraAxisObjectCoords = np.dot(cameraToObjectCoords, cameraAxis)
                self.environment.model = hm.rotation(self.environment.model, math.degrees(angle), cameraAxisObjectCoords)
                self.oldMouseX = self.curMouseX
                self.oldMouseY = self.curMouseY

    def calcArcBallVector(self, mouseX, mouseY):
        vec = np.array([float(mouseX) / self.windowWidth * 2 - 1.0, float(mouseY) / self.windowHeight * 2 - 1.0, 0], dtype = np.float32)
        vec[1] = -vec[1]
        distSquared = vec[0] * vec[0] + vec[1] * vec[1]
        if distSquared <= 1.0:
            vec[2] = math.sqrt(1.0 - distSquared)
        else:
            vec *= 1 / np.linalg.norm(vec)
        return vec

    def readData(self, fileName):
        f = open(fileName, 'r')
        for line in f: 
            s = line.split()
            position = np.array([s[0], s[1], s[2]], dtype = np.float32)
            self.environment.addDataPoint(position)
    
