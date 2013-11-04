import sys
import os
import time
import logging.config

from vision.util import isImageFile

class Context:
  """Application context class to store global data, configuration and objects."""
  
  @classmethod
  def createInstance(cls, *args, **kwargs):
    cls.instance = Context(*args, **kwargs)
    return cls.instance
  
  @classmethod
  def getInstance(cls):
    try:
      return cls.instance
    except AttributeError:
      raise Exception("Context.getInstance(): Called before context was created.")
  
  def __init__(self, argv=sys.argv):
    # * Ensure singleton
    if hasattr(self.__class__, 'instance'):
      raise Exception("Context.__init__(): Singleton instance already exists!")
    
    # * Obtain resource path and other parameters
    # TODO Start using optparse/argparse instead of positional arguments
    self.resPath = os.path.abspath(argv[1] if len(argv) > 1 else os.path.join("..", "res"))  # NOTE only absolute path seems to work properly
    self.gui = '--gui' in argv
    self.debug = '--debug' in argv
    
    # * Setup logging (before any other object is initialized that obtains a logger)
    # ** Load configuration from file
    logConfigFile = self.getResourcePath("config", "logging.conf")  # TODO make log config filename an optional argument
    os.chdir(os.path.dirname(logConfigFile))  # change to log config file's directory (it contains relative paths)
    logging.config.fileConfig(logConfigFile)  # load configuration
    os.chdir(sys.path[0])  # change back to current script's directory
    # ** Tweak root logger configuration based on command-line arguments
    if self.debug and logging.getLogger().getEffectiveLevel() > logging.DEBUG:
      logging.getLogger().setLevel(logging.DEBUG)
    elif not self.debug and logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
      logging.getLogger().setLevel(logging.INFO)  # one level above DEBUG
      # NOTE Logging level order: DEBUG < INFO < WARN < ERROR < CRITICAL
    
    # * Obtain camera device no. or input video/image filename
    self.cameraDevice = 0  # NOTE A default video/image filename can be specified here, but isVideo/isImage must be set accordingly
    self.isVideo = False
    self.isImage = False
    if len(sys.argv) > 2:
      try:
        self.cameraDevice = int(sys.argv[2])  # works if sys.argv[2] is an int (device no.)
        self.isVideo = False
        self.isImage = False
      except ValueError:
        self.cameraDevice = os.path.abspath(sys.argv[2])  # fallback: treat sys.argv[2] as string (filename)
        if isImageFile(self.cameraDevice):
          self.isVideo = False
          self.isImage = True
        else:
          self.isVideo = True
          self.isImage = False
    
    # * Timing
    self.resetTime()
  
  def resetTime(self):
    self.timeStart = time.time()
    self.timeNow = 0.0
    self.isPaused = False
    self.timePaused = self.timeStart
  
  def update(self):
    self.timeNow = time.time() - self.timeStart
  
  def pause(self):
    self.timePaused = time.time()
    self.isPaused = True
  
  def resume(self):
    self.timeStart += time.time() - self.timePaused
    self.isPaused = False
  
  def getResourcePath(self, subdir, filename):
    return os.path.abspath(os.path.join(self.resPath, subdir, filename))
