import numpy as np
import xml.etree.ElementTree as ET

from Component import Component

class Material(Component):
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    return Material(
      np.float32(eval(xmlElement.find('color').text)),
      actor)
    # NOTE this use of eval() is unsafe
    # TODO Catch missing subelement exception, use default value
  
  def __init__(self, color=np.float32([0.0, 0.0, 0.0]), actor=None):
    Component.__init__(self, actor)
    
    self.color = color
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      ET.SubElement(xmlElement, 'color', text=str(self.color))
      # NOTE Should we use repr() instead of str()?
  
  def __str__(self):
      return "Material: { color: " + str(self.color) + "}"
