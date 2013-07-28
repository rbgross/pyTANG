import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

from Actor import Actor
from Mesh import Mesh, EmptyMesh

class ActorFactory:
    def __init__(self, renderer, environment):
        self.renderer = renderer
        self.environment = environment
        self.meshes = dict()  # to cache meshes for reuse (TODO: move this cache to Mesh, as a static variable)
        self.loadResources()

    def loadResources(self):
        self.meshes['Empty'] = EmptyMesh()  # special empty mesh to be used as a placeholder, and for empty actors
        for filename in ['Cube.obj', 'CubeEdge.obj', 'SmallSphere.obj', 'Dragon.obj', 'TinySphere.obj']:
            self.getMesh(filename)  # ensures this mesh is loaded

    def getMesh(self, filename):
        if not filename in self.meshes:
            self.meshes[filename] = Mesh(os.path.abspath(os.path.join(self.renderer.resPath, 'models', filename)))
        return self.meshes[filename]

    def makeEmpty(self):
        return Actor(self.renderer, self.environment, self.meshes['Empty'])

    def makeCube(self):
        cube = Actor(self.renderer, self.environment, self.meshes['Cube.obj'])
        return cube

    def makeEdge(self):
        edge = Actor(self.renderer, self.environment, self.meshes['CubeEdge.obj'])
        return edge

    def makeDataPoint(self):
        dataPoint = Actor(self.renderer, self.environment, self.meshes['SmallSphere.obj'])
        return dataPoint

    def makeDragon(self):
        dragon = Actor(self.renderer, self.environment, self.meshes['Dragon.obj'])
        return dragon
