import numpy as np
import xml.etree.ElementTree as ET

from Component import Component

class Transform(Component):
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    return Transform(
      np.float32(eval(xmlElement.find('position').text)),
      np.float32(eval(xmlElement.find('rotation').text)),
      np.float32(eval(xmlElement.find('scale').text)),
      actor)
    # NOTE this use of eval() is unsafe
    # TODO Catch missing subelement exceptions, use default values
  
  def __init__(self, position=np.float32([0.0, 0.0, 0.0]), rotation=np.float32([0.0, 0.0, 0.0]), scale=np.float32([1.0, 1.0, 1.0]), actor=None):
    Component.__init__(self, actor)
    
    self.position = position
    self.rotation = rotation
    self.scale = scale
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      ET.SubElement(xmlElement, 'position', text=str(self.position))
      ET.SubElement(xmlElement, 'rotation', text=str(self.rotation))
      ET.SubElement(xmlElement, 'scale', text=str(self.scale))
      # NOTE Should we use repr() instead of str()?
  
  def __str__(self):
      return "Transform: {\n" + \
             "  position: " + str(self.position) + ",\n" + \
             "  rotation: " + str(self.rotation) + ",\n" + \
             "  scale: " + str(self.scale) + "\n" + \
             "}"