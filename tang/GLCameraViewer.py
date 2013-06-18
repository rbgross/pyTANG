# Python imports
import sys

# OpenCV imports
import numpy as np
import cv2
import cv2.cv as cv

# OpenGL imports
from OpenGL.GL import *
from OpenGL.arrays import vbo
import glfw

# Globals
# NOTE: glBlitFramebuffer() doesn't play well if source and destination sizes are different, so keep these same
windowWidth, windowHeight = (640, 480)
cameraWidth, cameraHeight = (640, 480)

class GLCameraViewer:
  """Simple OpenCV frame processor that renders live camera images using OpenGL 3.2 Core Profile."""
  
  def __init__(self, camera):
    self.camera = camera
    # * Set camera frame size
    #_, self.imageIn = self.camera.read()  # pre-grab
    self.camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, cameraWidth)
    self.camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, cameraHeight)
    _, self.imageIn = self.camera.read()  # post-grab
    print "GLCameraViewer.__init__(): Camera size: {}x{}".format(int(self.camera.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
    self.imageSize = (self.imageIn.shape[1], self.imageIn.shape[0])
    self.imageWidth, self.imageHeight = self.imageSize
    print "GLCameraViewer.__init__(): Image size : {}x{}".format(self.imageWidth, self.imageHeight)
    # NOTE: If camera frame size is not one supported by the hardware, grabbed images are scaled to desired size, discarding aspect-ratio
    
    # * Initialize OpenGL texture and framebuffer used to render camera images
    # NOTE: A valid OpenGL context must available at this point
    self.texOutId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, self.texOutId)
    #glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # image data is not padded (?)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)  # GL_NEAREST
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)  # GL_NEAREST
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)  # GL_REPEAT
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)  # GL_REPEAT
    
    self.framebufferId = glGenFramebuffers(1)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.framebufferId)
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texOutId, 0)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
    
    # * Other parameters
    # ** Create a random rectangular lens for some sample image processing
    self.lensWidth, self.lensHeight = (128, 128)
    self.lensX, self.lensY = (np.random.randint(self.imageWidth - 100), np.random.randint(self.imageHeight - 100))
    self.lensVelX, self.lensVelY = (np.random.randint(-3, 3) * 2, np.random.randint(-3, 3) * 2)
    if self.lensVelX == 0: self.lensVelX = 2
    if self.lensVelY == 0: self.lensVelY = 2
    
  def capture(self):
    _, self.imageIn = self.camera.read()
  
  def process(self):
    # Some sample image processing - just for fun!
    #self.imageOut = self.imageIn  # shallow copy
    self.imageOut = self.imageIn.copy()  # deep copy
    
    imageHSV = cv2.cvtColor(self.imageIn, cv2.COLOR_BGR2HSV)  # convert to HSV
    self.imageOut[self.lensY:self.lensY+self.lensHeight, self.lensX:self.lensX+self.lensWidth] = imageHSV[self.lensY:self.lensY+self.lensHeight, self.lensX:self.lensX+self.lensWidth]  # copy HSV image to within lens rectangle
    # Move lens and make it bounce
    self.lensX += self.lensVelX
    self.lensY += self.lensVelY
    if self.lensX <= 0:
      self.lensX = 0
      self.lensVelX = -self.lensVelX
    elif self.lensX >= (self.imageWidth - self.lensWidth - 1):
      self.lensX = self.imageWidth - self.lensWidth - 1
      self.lensVelX = -self.lensVelX
    if self.lensY <= 0:
      self.lensY = 0
      self.lensVelY = -self.lensVelY
    elif self.lensY >= (self.imageHeight - self.lensHeight - 1):
      self.lensY = self.imageHeight - self.lensHeight - 1
      self.lensVelY = -self.lensVelY
  
  def render(self):
    glBindTexture(GL_TEXTURE_2D, self.texOutId)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.imageWidth, self.imageHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, self.imageOut)
    
    glBindFramebuffer(GL_READ_FRAMEBUFFER, self.framebufferId)
    glBlitFramebuffer(
      0, 0, self.imageWidth, self.imageHeight,  # source rect
      0, windowHeight, windowWidth, 0,          # destination rect (NOTE: Y is flipped)
      GL_COLOR_BUFFER_BIT, GL_LINEAR)
    
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)
  
  def cleanUp(self):
    if bool(glDeleteFramebuffers):
      glDeleteFramebuffers([self.framebufferId])
    glDeleteTextures([self.texOutId])


def main():
  # * Initialize graphics subsystem
  initGraphics()
  
  # * Open camera and create GLCameraViewer instance
  camera = cv2.VideoCapture(0)  # NOTE: Live camera can be substituted with recorded video here
  cameraViewer = GLCameraViewer(camera)
  
  # * Main GLFW loop
  while glfw.GetWindowParam(glfw.OPENED):
    # ** Handle events
    glfw.PollEvents()
    if glfw.GetKey(glfw.KEY_ESC):
      break
    
    # ** Run cameraViewer through one iteration of processing
    cameraViewer.capture()
    cameraViewer.process()
    
    # ** Clear current output and render
    glClear(GL_COLOR_BUFFER_BIT)
    cameraViewer.render()
    
    # ** Present rendered output
    glfw.SwapBuffers()
  
  # * Clean up
  cameraViewer.cleanUp()
  camera.release()
  glfw.Terminate()


def initGraphics():
  # * Initialize GLFW and create OpenGL context
  glfw.Init()
  glfw.OpenWindowHint(glfw.FSAA_SAMPLES, 4)
  glfw.OpenWindowHint(glfw.OPENGL_VERSION_MAJOR, 3)
  glfw.OpenWindowHint(glfw.OPENGL_VERSION_MINOR, 2)
  glfw.OpenWindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
  #glfw.OpenWindowHint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
  glfw.OpenWindowHint(glfw.WINDOW_NO_RESIZE, GL_TRUE)
  glfw.OpenWindow(windowWidth, windowHeight, 0, 0, 0, 0, 0, 0, glfw.WINDOW)
  glfw.SetWindowTitle("GL Camera Viewer".format(OpenGL=glGetString(GL_VERSION), GLSL=glGetString(GL_SHADING_LANGUAGE_VERSION)))
  glfw.Disable(glfw.AUTO_POLL_EVENTS)
  glfw.Enable(glfw.KEY_REPEAT)
  
  print "initGraphics(): OpenGL version:", glGetString(GL_VERSION)  # glfw.GetGLVersion()
  print "initGraphics(): GLSL version  :", glGetString(GL_SHADING_LANGUAGE_VERSION)
  print "initGraphics(): Renderer      :", glGetString(GL_RENDERER)
  print "initGraphics(): Vendor        :", glGetString(GL_VENDOR)
  print "initGraphics(): Window size   : {}x{}".format(windowWidth, windowHeight)
  
  # * Initialize OpenGL parameters
  glClearColor(0.0, 0.0, 0.0, 1.0)


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


if __name__ == '__main__':
  main()
