import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

class Actor:
    def __init__(self, renderer, environment, mesh):
        self.renderer = renderer
        self.environment = environment
        
        # TODO create components dictionary to store all actor components by name
        # TODO Group separate Transform elements into a single component class, just like Mesh; similarly for Material
        
        # Mesh component
        self.mesh = mesh
        
        # Transform component
        self.position = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        self.scale = np.array([1.0, 1.0, 1.0], dtype = np.float32)
        
        # Material component
        self.color = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        
        # Child actors
        self.children = []
    
    def draw(self):
        model = hm.translation(self.environment.model, self.position)  # TODO make transform relative to parent, not absolute
        model = hm.rotation(model, self.rotation[0], [1, 0, 0])
        model = hm.rotation(model, self.rotation[1], [0, 1, 0])
        model = hm.rotation(model, self.rotation[2], [0, 0, 1])
        self.renderer.setModel(model)
        self.renderer.setDiffCol(self.color)
        self.mesh.draw()
        
        for child in self.children:
            child.draw() # TODO implement ability to show/hide actors and/or children
    
    def __str__(self):
        return self.toString()
    
    def toString(self, indent=""):
        out = indent + "Actor: {\n"
        out += indent + "  components: {\n"
        out += indent + "    " + str(self.mesh) + ",\n"
        out += indent + "    Transform: {\n"
        out += indent + "      position: " + str(self.position) + ",\n"
        out += indent + "      rotation: " + str(self.rotation) + ",\n"
        out += indent + "      scale: " + str(self.scale) + "\n"
        out += indent + "    },\n"
        out += indent + "    Material: {\n"
        out += indent + "      color: " + str(self.color) + "\n"
        out += indent + "    }\n"
        out += indent + "  },\n"
        out += indent + "  children: {\n"
        for child in self.children:
            out += child.toString(indent + "    ")
        out += indent + "  }\n"
        out += indent + "}\n"
        
        return out
