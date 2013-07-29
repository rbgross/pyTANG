import xml.etree.ElementTree as ET

class Component:
  """Base class for all Actor components."""
  
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    """Create a Component instance from XML element."""
    # NOTE Subclasses should override this to extract relevant properties from xmlElement
    # TODO Delegate component creation to appropriate subclass based on XML tag, return None when invalid
    return Component(actor)
  
  def __init__(self, actor=None):
    self.actor = actor  # needed to maintain a link to containing actor
    # NOTE Subclasses should call Component.__init__(self, actor) first
  
  def toXMLElement(self):
    """Convert this instance to an XML element."""
    # NOTE Subclasses should call Component.toXMLElement(self) to obtain
    #   base node and then add further attributes and sub-elements
    return ET.Element(self.__class__.__name__)
  
  def __str__(self):
    """Return a brief string representation of this component object."""
    return "Component: { }"