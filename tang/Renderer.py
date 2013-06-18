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

from Shader import Shader

class Renderer:
    def __init__(self, resPath=None):
        self.resPath = resPath if resPath is not None else os.getcwd()  # default to current directory if None passed
        
        self.windowWidth = 640
        self.windowHeight = 480

        self.initialize()

        self.colorShader = Shader(os.path.abspath(os.path.join(self.resPath, 'shaders', 'ADS.vert')), os.path.abspath(os.path.join(self.resPath, 'shaders', 'ADS.frag')))

        self.cameraMatrix = np.dot(hm.perspective(hm.identity(), 45, float(self.windowWidth) / self.windowHeight, 0.1, 1000.0), hm.lookat(hm.identity(), np.array([0.0, 0.0, 55.0, 1.0], dtype = np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32)))
        self.setCameraMatrix(self.cameraMatrix)

    def setCameraMatrix(self, cameraMatrix):
        self.colorShader.setUniformMat4('view', hm.identity())
        self.colorShader.setUniformMat4('proj', cameraMatrix)

    def setModel(self, model):
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
        glfw.OpenWindow(self.windowWidth, self.windowHeight, 0, 0, 0, 0, 0, 0, glfw.WINDOW)
        glfw.SetWindowTitle("TANG")

        glEnable(GL_DEPTH_TEST)
	glEnable(GL_CULL_FACE)

    def readData(self, fileName):
        f = open(fileName, 'r')
        for line in f: 
            s = line.split()
            position = np.array([s[0], s[1], s[2]], dtype = np.float32)
            self.environment.addDataPoint(position)
    
