import time
import os
from OpenGL.GL import *
import glfw
from ctypes import *
import sys

def main():
    #Initialize the OpenGL context and window
    initialize()

    vertexSource = '''
    #version 150\n
    in vec2 position;
    in vec3 color;
    out vec3 Color;
    void main() {
        Color = color;
        gl_Position = vec4( position, 0.0, 1.0 );
    }'''

    fragmentSource = '''
    #version 150\n
    in vec3 Color;
    out vec4 outColor;
    void main() {
        outColor = vec4( Color, 1.0 );
    }'''

    #Create and compile the vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertexShader, vertexSource)
    glCompileShader(vertexShader)
    printShaderInfoLog( vertexShader )

    #Create and compile the fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragmentShader, fragmentSource)
    glCompileShader(fragmentShader)
    printShaderInfoLog( fragmentShader )

    #Link the vertex and fragment shader into a shader program
    shaderProgram = glCreateProgram()
    glAttachShader(shaderProgram, vertexShader)
    glAttachShader(shaderProgram, fragmentShader)
    glBindFragDataLocation( shaderProgram, 0, "outColor" )
    glLinkProgram( shaderProgram )
    printProgramInfoLog( shaderProgram )
    glUseProgram( shaderProgram )

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    #CAN YOU USE LISTS HERE?
    vbo = glGenBuffers(1)    
    vertices = [-0.5,  0.5, 1.0, 0.0, 0.0, #Top-left
		 0.5,  0.5, 0.0, 1.0, 0.0, #Top-right
		 0.5, -0.5, 0.0, 0.0, 1.0, #Bottom-right
		-0.5, -0.5, 1.0, 1.0, 1.0] #Bottom-left
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices)*4, (c_float*len(vertices))(*vertices), GL_STATIC_DRAW)
    
    ebo = glGenBuffers(1)
    elements = [0, 1, 2, 2, 3, 0]
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(elements)*4, (c_uint*len(elements))(*elements), GL_STATIC_DRAW)

    posAttrib = glGetAttribLocation(shaderProgram, "position")
    glEnableVertexAttribArray(posAttrib)
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 5*4, vbo+4*0)

    colAttrib = glGetAttribLocation(shaderProgram, "color")
    glEnableVertexAttribArray(colAttrib)
    glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 5*4, vbo+4*2)
    
    while glfw.GetWindowParam(glfw.OPENED):
        if glfw.GetKey(glfw.KEY_ESC):
            break

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glDrawElements( GL_TRIANGLES, 6, GL_UNSIGNED_INT, None )
        
        glfw.SwapBuffers()

    glfw.Terminate()
    return

def initialize():
    glfw.Init()
    glfw.OpenWindowHint(glfw.FSAA_SAMPLES, 4)
    glfw.OpenWindowHint(glfw.OPENGL_VERSION_MAJOR, 3)
    glfw.OpenWindowHint(glfw.OPENGL_VERSION_MINOR, 2)
    glfw.OpenWindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.OpenWindowHint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.OpenWindowHint(glfw.WINDOW_NO_RESIZE, GL_TRUE)
    glfw.OpenWindow(800, 600, 0, 0, 0, 0, 0, 0, glfw.WINDOW)
    glfw.SetWindowTitle("OpenGL")
    #glew init here?

def printShaderInfoLog(obj):
    infoLogLength = glGetShaderiv(obj, GL_INFO_LOG_LENGTH)

    if infoLogLength > 1:
        info = glGetShaderInfoLog(obj)
        print >> sys.stderr, info

def printProgramInfoLog(obj):
    infoLogLength = glGetProgramiv(obj, GL_INFO_LOG_LENGTH)

    if infoLogLength > 1:
        info = glGetProgramInfoLog(obj)
        print >> sys.stderr, info   

if __name__ == '__main__': main()
