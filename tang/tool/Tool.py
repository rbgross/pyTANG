import logging

from Context import Context

class Tool:
  """Base class for all tools."""
  
  def __init__(self):
    """Initialize tool-specific behavior here."""
    self.context = Context.getInstance()
    self.logger = logging.getLogger(__name__)
    self.active = False
  
  def activate(self):
    self.active = True
    self.logger.info("[{}] Activated".format(self.__class__.__name__))
  
  def deactivate(self):
    self.active = False
    self.logger.info("[{}] Deactivated".format(self.__class__.__name__))
  
  def toggle(self):
    if self.active:
      self.deactivate()
    else:
      self.activate()
  
  def update(self):
    """Perform tool-specific updates here, including actor transform in scene."""
    pass