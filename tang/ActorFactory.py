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
from component.Mesh import Mesh
from component.Transform import Transform
from component.Material import Material

class ActorFactory:
    def __init__(self, renderer, scene):
        self.renderer = renderer
        self.scene = scene
        self.loadResources()

    def loadResources(self):
        for src in ['Cube.obj', 'CubeEdge.obj', 'SmallSphere.obj', 'Dragon.obj', 'TinySphere.obj']:
            Mesh.getMesh(src)  # ensures this mesh is loaded

    def makeEmpty(self):
        """Create an Actor with no components; useful for building up an actor from scratch."""
        return Actor(self.renderer, self.scene)

    def makeDefault(self, meshSrc="Empty"):
        """Create an Actor with a default set of components, and specified mesh."""
        actor = Actor(self.renderer, self.scene)
        actor.components['Mesh'] = Mesh.getMesh(meshSrc)  # NOTE Meshes are currently shared, therefore not linked to individual actors
        actor.components['Transform'] = Transform(actor=actor)
        actor.components['Material'] = Material(actor=actor)
        return actor

    def makeCube(self):
        return self.makeDefault(meshSrc='Cube.obj')

    def makeEdge(self):
        return self.makeDefault(meshSrc='CubeEdge.obj')

    def makeDataPoint(self):
        return self.makeDefault(meshSrc='SmallSphere.obj')

    def makeDragon(self):
        return self.makeDefault(meshSrc='Dragon.obj')
