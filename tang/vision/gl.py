"""Vision-related functionality that is dependent on OpenGL."""

# OpenCV imports
import cv2
import cv2.cv as cv

# GL imports
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GL.framebufferobjects import *

# Custom imports
from base import FrameProcessor

class FrameProcessorGL(FrameProcessor):
  """A FrameProcessor that is able to render its output image using OpenGL."""
  
  def __init__(self, options):
    FrameProcessor.__init__(self, options)
    self.windowWidth = options.get('windowWidth', 640)
    self.windowHeight = options.get('windowHeight', 480)
    
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
  
  def initialize(self, image, timeNow):
    self.image = image
    self.imageSize = (self.image.shape[1], self.image.shape[0])
    self.logger.debug("Image size: {}x{}".format(self.imageSize[0], self.imageSize[1]))
    self.imageWidth, self.imageHeight = self.imageSize  # will be used frequently
    self.imageOut = self.image.copy()
    self.imageRendered = False
    self.active = True
  
  def process(self, image, timeNow):
    self.image = image
    
    if self.imageRendered and self.image is not None:
      self.imageOut = self.image.copy()  # only copy if the last one had been used (at least once)
      self.imageRendered = False
    
    return self.imageOut
  
  def render(self):
    try:
      imageToRender = cv2.flip(self.imageOut, 0)  # flip OpenCV image vertically to match OpenGL convention (necessary on Windows because of glBlitFramebuffer problem; avoid if possible)
      
      glBindTexture(GL_TEXTURE_2D, self.texOutId)
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.imageWidth, self.imageHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, imageToRender)
      
      glBindFramebuffer(GL_READ_FRAMEBUFFER, self.framebufferId)
      # TODO Fix glBlitFramebuffer() problem on Windows, or use an alternate method to draw CV image
      #glBlitFramebuffer(
      #  0, 0, self.imageWidth, self.imageHeight,    # source rect
      #  0, self.windowHeight, self.windowWidth, 0,  # destination rect (NOTE: Y is flipped)
      #  GL_COLOR_BUFFER_BIT, GL_LINEAR)  # NOTE trying to flip while blitting doesn't work on Windows
      
      glBlitFramebuffer(
        0, 0, self.imageWidth,  self.imageHeight,   # source rect
        0, 0, self.windowWidth, self.windowHeight,  # destination rect
        GL_COLOR_BUFFER_BIT, GL_LINEAR)  # direct blit without any flipping works on Windows
      
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
      glBindTexture(GL_TEXTURE_2D, 0)
      
      self.imageRendered = True
    except GLError as e:
      self.logger.error(repr(e))  # print str(e) for more details, or don't catch this error to break out
      if self.gui and self.debug:
        cv2.imshow("{}".format(self.__class__.__name__), self.imageOut)  # optional, so that we can verify OpenCV is working
  
  def cleanUp(self):
    self.logger.debug("Cleaning up...")
    try:
      if bool(glDeleteFramebuffers):
        glDeleteFramebuffers([self.framebufferId])
      glDeleteTextures([self.texOutId])
    except AttributeError as e:
      self.logger.error("Can't clean up! Not initialized properly? Error: %s", e)
  
  def __del__(self):
    self.cleanUp()
