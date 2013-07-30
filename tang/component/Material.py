import numpy as np
import xml.etree.ElementTree as ET

from Component import Component

class Material(Component):
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    color = xmlElement.get('color')
    return Material(
      np.fromstring(color, dtype=np.float32, sep=' ') if color is not None else np.float32([0.0, 0.0, 0.0]),
      actor)
  
  def __init__(self, color=np.float32([0.0, 0.0, 0.0]), actor=None):
    Component.__init__(self, actor)
    self.color = color
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      xmlElement.set('color', str(self.color).strip('[ ]'))
      return xmlElement
  
  def __str__(self):
      return "Material: { color: " + str(self.color) + "}"
