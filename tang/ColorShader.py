import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm

class ColorShader:
    def __init__(self):
        vertexSource = '''
        #version 150\n
	in vec3 position;
	in vec3 normal;
	out vec3 eyeNormal;
	out vec3 eyePosition;
	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 proj;
	void main() {
            eyeNormal = mat3( view * model ) * normal;
	    eyePosition = vec3( view * model * vec4( position, 1.0f ) );
	    gl_Position = proj * view * model * vec4( position, 1.0f );
	}'''

        fragmentSource = '''
        #version 150\n
        in vec3 eyePosition;
        in vec3 eyeNormal;
        out vec4 outColor;
        uniform mat4 view;
        uniform vec4 lightPosition;
        uniform vec3 diffuseColor;
        void main() {
            vec4 eyeLightPosition = view * lightPosition;
            vec3 normal = normalize( eyeNormal );
            vec3 toLightDir = normalize( eyeLightPosition.xyz - eyeLightPosition.w * eyePosition );
            vec3 lightIntensity = vec3( 1.0f, 1.0f, 1.0f );
            vec3 ambientColor = 0.2f * diffuseColor;
            vec3 ambient = ambientColor;
            vec3 diffuse = diffuseColor * max ( dot( normal, toLightDir ), 0.0f );
            outColor = vec4( lightIntensity * ( ambient + diffuse ), 1.0f );
        }'''

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
	glDeleteShader( vertexShader );
	glDeleteShader( fragmentShader );

    def setModel(self, model):
        glUniformMatrix4fv( glGetUniformLocation( self.shaderProgram, "model" ), 1, GL_TRUE, model)
            
    def setView(self, view):
        glUniformMatrix4fv( glGetUniformLocation( self.shaderProgram, "view" ), 1, GL_TRUE, view)

    def setProj(self, proj):
        glUniformMatrix4fv( glGetUniformLocation( self.shaderProgram, "proj" ), 1, GL_TRUE, proj)

    def setLightPos(self, lightPos):
        glUniform4fv( glGetUniformLocation( self.shaderProgram, "lightPosition" ), 1, lightPos)

    def setDiffCol(self, diffCol):
        glUniform3fv( glGetUniformLocation( self.shaderProgram, "diffuseColor" ), 1, diffCol)

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
