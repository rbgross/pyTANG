import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np

from Environment import Environment
from ColorShader import ColorShader

class Renderer:
    def __init__(self):
        self.windowWidth = 640
        self.windowHeight = 480

        self.initialize()

        self.colorShader = ColorShader()

        self.view = glm.lookAt(glm.vec3(0.0, 0.0, 55.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        self.setView(self.view)

        self.proj = glm.perspective(45.0, self.windowWidth / self.windowHeight, 0.1, 1000.0)
        self.setProj(self.proj)

        self.environment = Environment(self)

        self.wheelPosition = glfw.GetMouseWheel()
        self.oldMouseX, self.oldMouseY = glfw.GetMousePos()
        self.curMouseX = self.oldMouseX
        self.curMouseY = self.oldMouseY
        self.hideCube = False
        self.leftPressed = False
        self.rightPressed = False

        self.readData('C:\\Users\\Ryan\\Game Tests\\Data.txt')

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
        return glfwGetWindowParam(glfw.OPENED)

    def draw(self):
        self.pollInput()
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.environment.draw()

        glfw.SwapBuffer()

    def initialize(self):
        glfw.Init()
        glfw.OpenWindowHint(glfw.FSAA_SAMPLES, 4)
        glfw.OpenWindowHint(glfw.OPENGL_VERSION_MAJOR, 3)
        glfw.OpenWindowHint(glfw.OPENGL_VERSION_MINOR, 2)
        glfw.OpenWindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.OpenWindowHint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.OpenWindowHint(glfw.WINDOW_NO_RESIZE, GL_TRUE)
        glfw.OpenWindow(800, 600, 0, 0, 0, 0, 0, 0, glfw.WINDOW)
        glfw.SetWindowTitle("TANG")

    def pollInput(self):
        tempWheelPosition = glfw.GetMouseWheel()
        if tempWheelPosition != self.wheelPosition:
            self.wheelPosition = tempWheelPosition
            self.setView(glm.lookAt(glm.vec3(0.0, 0.0, 55.0 - self.wheelPosition), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0)))

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
            self.curMouseX, self.curMouseY = glfw.GetMousePos()
            if self.curMouseX != self.oldMouseX or self.curMouseY != self.oldMouseY:
                oldVec = self.calcArcBallVector(self.oldMouseX, self.oldMouseY)
                curVec = self.calcArcBallVector(self.curMouseX, self.curMouseY)
                angle = math.acos(min(1.0, glm.dot(oldVec, curVec)))
                cameraAxis = glm.cross(oldVec, curVec)
                cameraToObjectCoords = glm.inverse(glm.mat3(self.view) * glm.mat3(self.environment.model))
                cameraToAxisObjectCoords = cameraToObjectCoords * cameraAxis
                self.environment.model = glm.rotate(self.environment.model, glm.degrees(angle), cameraAxisObjectCoords)
                self.oldMouseX = self.curMouseX
                self.oldMouseY = self.curMouseY

    def calcArcBallVector(self, mouseX, mouseY):
        vec = glm.vec3(mouseX / self.windowWidth * 2 - 1.0, mouseY / self.windowHeight * 2 - 1.0, 0)
        vec.y = -vec.y
        distSquared = vec.x * vec.x + vec.y * vec.y
        if distSquared <= 1.0:
            vec.z = math.sqrt(1.0 - distSquared)
        else:
            vec = glm.normalize(vec)
        return vec

    def readData(self, fileName):
        f = open(fileName, 'r')
        for line in f: 
            s = line.split()
            position = glm.vec3(s[0], s[1], s[2])
            self.environment.addDataPoint(position)
    
