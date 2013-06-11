import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

class Mesh:
    def __init__(self, fileName):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.loadModel(fileName)

        self.vbo = VBO(self.meshData, GL_STATIC_DRAW)
        self.vbo.bind()

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*4, self.vbo+0)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*4, self.vbo+12)
        
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.elements)*4, self.elements, GL_STATIC_DRAW)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.elements.size, GL_UNSIGNED_INT, None )

    def loadModel(self, fileName):
        f = open(fileName, 'r')
        self.meshData = []
        self.elements = []
        vertices = []
        normals = []
        texCoords = []

        for line in f:
            s = line.split()
            if len(s) > 0:
                if s[0] == 'v':
                    v = np.array([s[1], s[2], s[3]], dtype = np.float32)
                    vertices.append(v)
                elif s[0] == 'vn':
                    v = np.array([s[1], s[2], s[3]], dtype = np.float32)
                    normals.append(v)
                elif s[0] == 'vt':
                    v = np.array([s[1], s[2]], dtype = np.float32)
                    texCoords.append(v)
                elif s[0] == 'f':
                    for i in xrange(1, 4):
                        l = s[i].split('/')
                        self.meshData.append(float(vertices[int(l[0]) - 1][0]))
                        self.meshData.append(float(vertices[int(l[0]) - 1][1]))
                        self.meshData.append(float(vertices[int(l[0]) - 1][2]))
                        self.meshData.append(float(normals[int(l[2]) - 1][0]))
                        self.meshData.append(float(normals[int(l[2]) - 1][1]))
                        self.meshData.append(float(normals[int(l[2]) - 1][2]))
                        self.elements.append(len(self.elements))

        self.meshData = np.array(self.meshData, dtype = np.float32)
        self.elements = np.array(self.elements, dtype = np.uint32)
                
        
