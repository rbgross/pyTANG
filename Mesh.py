import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np

class Cube:
    def __init__(self, fileName):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.loadModel(fileName)

        vbo = VBO(self.meshData, GL_STATIC_DRAW)
        vbo.bind()

        posAttrib = glGetAttribLocation(shaderProgram, "position")
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 6*4, vbo+0)

        normAttrib = glGetAttribLocation(shaderProgram, "normal")
        glEnableVertexAttribArray(normAttrib)
        glVertexAttribPointer(normAttrib, 3, GL_FLOAT, GL_FALSE, 6*4, vbo+12)
        
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.elements)*4, self.elements, GL_STATIC_DRAW)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.elements.size, GL_UNSIGNED_INT, None )

    def loadModel(self, fileName):
        f = open(fileName, 'r')
        vertices = []
        normals = []
        texCoords = []

        for line in f:
            s = line.split()
            if s[0] == 'v':
                v = glm.vec3(s[1], s[2], s[3])
                vertices.append(v)
            elif s[0] == 'vn':
                v = glm.vec3(s[1], s[2], s[3])
                normals.append(v)
            elif s[0] == 'vt':
                v = glm.vec3(s[1],s[2])
                texCoords.append(v)
            elif s[0] == 'f':
                for i in xrange(1, 4):
                    l = s[i].split('/')
                    self.meshData.append(vertices[l[0] - 1].x)
                    self.meshData.append(vertices[l[0] - 1].y)
                    self.meshData.append(vertices[l[0] - 1].z)
                    self.meshData.append(normals[l[2] - 1].x)
                    self.meshData.append(normals[l[2] - 1].y)
                    self.meshData.append(normals[l[2] - 1].z)
                    self.elements.append(len(self.elements))
                
        
