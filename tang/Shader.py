import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

from Context import Context

class Shader:
    def __init__(self, vertexFileName, fragmentFileName):
        #Read in the vertex shader from file
        with open (vertexFileName, "r") as f:
            vertexSource = f.read()
        f.close()

        #Read in the fragment shader from file
        with open (fragmentFileName, "r") as f:
            fragmentSource = f.read()
        f.close()
        
        # OpenGL version-dependent code (NOTE shader source must be version 150)
        if Context.getInstance().GLSL_version_string != "150":
          print "Shader.__init__(): Changing shader source version to {}".format(Context.getInstance().GLSL_version_string)
          vertexSource = vertexSource.replace("150", Context.getInstance().GLSL_version_string, 1)
          fragmentSource = fragmentSource.replace("150", Context.getInstance().GLSL_version_string, 1)
        # TODO put special version placeholder in shader source files instead of "150"
        
        #Create and compile the vertex shader
        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertexSource)
        glCompileShader(vertexShader)
        self.printShaderInfoLog( vertexShader )

        #Create and compile the fragment shader
        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragmentSource)
        glCompileShader(fragmentShader)
        self.printShaderInfoLog( fragmentShader )

        #Link the vertex and fragment shader into a shader program
        self.shaderProgram = glCreateProgram()
        glAttachShader(self.shaderProgram, vertexShader)
        glAttachShader(self.shaderProgram, fragmentShader)

        glBindAttribLocation( self.shaderProgram, 0, "position")
        glBindAttribLocation( self.shaderProgram, 1, "normal")
        glBindFragDataLocation(self.shaderProgram, 0, "outColor")

        glLinkProgram(self.shaderProgram)
        self.printProgramInfoLog(self.shaderProgram)
        glUseProgram(self.shaderProgram)

        #Flag the vertex and fragment shaders for deletion when not in use
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

    def setUniformMat4(self, name, value):
        glUniformMatrix4fv(glGetUniformLocation(self.shaderProgram, name), 1, GL_TRUE, value)
            
    def setUniformMat3(self, name, value):
        glUniformMatrix3fv(glGetUniformLocation(self.shaderProgram, name), 1, GL_TRUE, value)

    def setUniformVec4(self, name, value):
        glUniform4fv(glGetUniformLocation(self.shaderProgram, name), 1, value)

    def setUniformVec3(self, name, value):
        glUniform3fv(glGetUniformLocation(self.shaderProgram, name), 1, value)

    def printShaderInfoLog(self, obj):
        infoLogLength = glGetShaderiv(obj, GL_INFO_LOG_LENGTH)
        if infoLogLength > 1:
            info = glGetShaderInfoLog(obj)
            print >> sys.stderr, info

    def printProgramInfoLog(self, obj):
        infoLogLength = glGetProgramiv(obj, GL_INFO_LOG_LENGTH)
        if infoLogLength > 1:
            info = glGetProgramInfoLog(obj)
            print >> sys.stderr, info   
