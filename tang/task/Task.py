import logging

from Context import Context

class Task:
  """Represents a particular task or activity that is meant to be carried out using the TANG system."""
  
  def __init__(self):
    """
    Initialize task-specific variables, load scene fragments, find tools.
    Derived classes must call through to Task.__init__(self).
    NOTE: Context must be initialized at this point.
    """
    self.context = Context.getInstance()
    self.logger = logging.getLogger(__name__)
    self.active = False
  
  def activate(self):
    self.active = True
    self.logger.info("[{}] Activated at time: {:.3f}".format(self.__class__.__name__, self.context.timeNow))
  
  def deactivate(self):
    self.active = False
    self.logger.info("[{}] Deactivated at time: {:.3f}".format(self.__class__.__name__, self.context.timeNow))
  
  def toggle(self):
    if self.active:
      self.deactivate()
    else:
      self.activate()
  
  def update(self):
    """Perform task-specific updates here."""
    pass
