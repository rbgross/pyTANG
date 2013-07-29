import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

from Component import Component

class Mesh(Component):
    resPath = ""
    meshes = dict()  # to cache meshes for reuse
    
    @classmethod
    def configure(cls, resPath):
        cls.resPath = resPath
        cls.meshes['Empty'] = EmptyMesh()  # special empty mesh to be used as a placeholder, and for empty actors
    
    @classmethod
    def fromXMLElement(cls, xmlElement, actor=None):
        try:
            return cls.getMesh(xmlElement.attrib['src'], actor)
        except KeyError as e:
            print "Mesh.fromXMLElement(): Error: Required attribute not found in XML element:", e
            return None
    
    @classmethod
    def getMesh(cls, src, actor=None):
        if not src in cls.meshes:
            cls.meshes[src] = Mesh(src, actor)
        return cls.meshes[src]
    
    def __init__(self, src, actor=None):
        Component.__init__(self, actor)

        # TODO Include a mesh name (e.g. 'Dragon') as ID as well as src (e.g. '../res/models/Dragon.obj')
        self.src = src
        self.filepath = os.path.abspath(os.path.join(Mesh.resPath, 'models', src))  # NOTE Mesh.configure() must be called prior to this to set resPath
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.loadModel(self.filepath)

        self.vbo = VBO(self.meshData, GL_STATIC_DRAW)
        self.vbo.bind()

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*4, self.vbo+0)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*4, self.vbo+12)
        
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.elements)*4, self.elements, GL_STATIC_DRAW)

    def loadModel(self, filename):
        f = open(filename, 'r')
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

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.elements.size, GL_UNSIGNED_INT, None)
    
    def toXMLElement(self):
        xmlElement = Component.toXMLElement(self)
        xmlElement.attrib['src'] = self.src
    
    def __str__(self):
        return "Mesh: { src: \"" + self.src + "\" }"


class EmptyMesh:
    """Special mesh class to use for empty actors."""
    
    def __init__(self, filename=None):
        pass
    
    def draw(self):
        pass
    
    def __str__(self):
        return "Mesh: { src: \"\" }"
