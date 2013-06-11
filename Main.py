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

def main():
    renderer = Renderer()
    while renderer.windowOpen():
        renderer.draw()

if __name__ == '__main__': main()

