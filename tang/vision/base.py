"""Base classes for OpenCV-based computer vision."""

# Python imports
import sys
import numpy as np
import logging

# CV imports
import cv2
import cv2.cv as cv

class FrameProcessor:
  """Processes a sequence of images (frames)."""
  
  def __init__(self, options):
    self.gui = options.get('gui', False)
    self.debug = options.get('debug', False)
    self.logger = logging.getLogger(self.__class__.__name__)
    self.active = False  # set to True once initialized
    # NOTE Subclasses should call FrameProcessor.__init__(self, options) first, then process options
  
  def initialize(self, image, timeNow):
    self.image = image
    self.imageSize = (self.image.shape[1], self.image.shape[0])
    self.logger.debug("Image size: {}x{}".format(self.imageSize[0], self.imageSize[1]))
    self.imageOut = self.image
    self.active = True  # must be set to true on initialize(), otherwise process() may not get called when in a pool
  
  def process(self, image, timeNow):
    self.image = image
    self.imageOut = self.image
    return self.imageOut
  
  def onKeyPress(self, key, keyChar=None):
    return True  # indicates if this event has been consumed by this processor


class FrameProcessorPool:
  """Abstract base class for all collections of FrameProcessors."""
  
  def getProcessorByType(self, processorType):
    raise NotImplementedError("FrameProcessorPool.getProcessorByType() is abstract; must be implmented by subclasses.")


class FrameProcessorPipeline(FrameProcessorPool):
  """An ordered pipeline of FrameProcessor instances with chained input-outputs."""
  
  def __init__(self, options, processorTypes):
    """Create a list of FrameProcessors given appropriate types."""
    self.options = options  # keep a copy in case processors need to be added in the future
    self.gui = self.options.get('gui', False)
    self.debug = self.options.get('debug', False)
    self.logger = logging.getLogger(self.__class__.__name__)
    
    self.options['pool'] = self  # for FrameProcessor types that are dependent on other types in this pool
    self.processors = []
    for processorType in processorTypes:
      if issubclass(processorType, FrameProcessor):
        processor = processorType(options)
        self.processors.append(processor)
        self.logger.debug("Added {0} instance.".format(processor.__class__.__name__))
      else:
        self.logger.warn("{0} is not a FrameProcessor; will not instantiate.".format(processorType.__name__))
  
  def initialize(self, image, timeNow):
    for processor in self.processors:
      processor.initialize(image, timeNow)
  
  def process(self, image, timeNow):
    for processor in self.processors:
      if processor.active:
        image = processor.process(image, timeNow)
        if self.gui and self.debug and imageOut is not None:  # show individual processor outputs only if gui and debug are true
          cv2.imshow("Output: {}".format(processor.__class__.__name__), image)
    return image
  
  def onKeyPress(self, key, keyChar=None):
    eventConsumed = False
    for processor in self.processors:
      if processor.active:
        eventConsumed = processor.onKeyPress(key, keyChar)  # pass along key-press to processor
        if eventConsumed:
          break  # break out of this for loop (no further processors get the key event)
    return eventConsumed
  
  def activateProcessors(self, processorTypes=None, active=True):  # if None, activate all
    for processor in self.processors:
      if processorTypes is None or processor.__class__ in processorTypes:
        processor.active = active
  
  def deactivateProcessors(self, processorTypes=None):  # if None, deactivate all
    self.activateProcessors(processorTypes, False)
  
  def getProcessorByType(self, processorType):
    """Returns the first processor found that is an instance of processorType."""
    for processor in self.processors:
      if isinstance(processor, processorType):
        return processor
    return None
  
  def __str__(self):
    desc = "[" + ", ".join(("" if processor.active else "~") + processor.__class__.__name__ for processor in self.processors) + "]"
    return desc
