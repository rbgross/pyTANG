import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

from Renderer import Renderer
from Environment import Environment
from Controller import Controller

def main():
    resPath = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "res"))  # NOTE only absolute path seems to work properly
    print "main(): Resource path:", resPath
    
    renderer = Renderer(resPath)
    environment = Environment(renderer)
    controller = Controller(environment)    
    
    while renderer.windowOpen():
        controller.pollInput()
        renderer.startDraw()
        #camera.draw()
        environment.draw()
        renderer.endDraw()

if __name__ == '__main__':
    main()
