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

from Context import Context
from Shader import Shader

class Renderer:
    def __init__(self):
        self.context = Context.getInstance()  # NOTE must be initialized
        
        self.windowWidth = 640
        self.windowHeight = 480

        self.initialize()

        self.colorShader = Shader(self.context.getResourcePath('shaders', 'ADS.vert'), self.context.getResourcePath('shaders', 'ADS.frag'))

        self.proj = hm.perspective(hm.identity(), 35, float(self.windowWidth) / self.windowHeight, 1.0, 1000.0)
        self.view = hm.lookat(hm.identity(), np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32), np.array([0.0, 0.0, 1.0, 1.0], dtype = np.float32), np.array([0.0, -1.0, 0.0, 1.0], dtype = np.float32))
        self.cameraMatrix = np.dot(self.proj, self.view)
        self.setCameraMatrix(self.cameraMatrix)

    def setCameraMatrix(self, cameraMatrix):
        self.colorShader.setUniformMat4('view', self.view)
        self.colorShader.setUniformMat4('proj', self.proj)

    def setModelMatrix(self, model):
        self.colorShader.setUniformMat4('model', model)

    def setLightPos(self, lightPos):
        self.colorShader.setUniformVec4('lightPosition', lightPos)

    def setDiffCol(self, diffCol):
        self.colorShader.setUniformVec3('diffuseColor', diffCol)

    def windowOpen(self):
        return glfw.GetWindowParam(glfw.OPENED)

    def startDraw(self):        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def endDraw(self):
        glfw.SwapBuffers()

    def initialize(self):
        glfw.Init()
        glfw.OpenWindowHint(glfw.FSAA_SAMPLES, 4)
        glfw.OpenWindowHint(glfw.OPENGL_VERSION_MAJOR, 3)
        glfw.OpenWindowHint(glfw.OPENGL_VERSION_MINOR, 2)
        glfw.OpenWindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.OpenWindowHint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.OpenWindowHint(glfw.WINDOW_NO_RESIZE, GL_TRUE)
        glfw.OpenWindow(self.windowWidth, self.windowHeight, 0, 0, 0, 0, 24, 8, glfw.WINDOW)
        glfw.SetWindowTitle("TANG")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
