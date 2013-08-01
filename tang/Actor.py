import time
import os
from OpenGL.arrays.vbo import VBO
from OpenGL.GL import *
import glfw
from ctypes import *
import sys
import numpy as np
import hommat as hm
import xml.etree.ElementTree as ET

from component.Component import Component

class Actor:
  @classmethod
  def fromXMLElement(cls, xmlElement, actorFactory):
    # Initialize empty actor object
    actor = actorFactory.makeEmpty()
    
    # TODO Extract common element attributes such as id, type?
    
    # Load actor's components
    components = xmlElement.find('components')
    for component in components:
      componentObj = Component.fromXMLElement(component, actor)
      if componentObj is not None:
        actor.components[componentObj.__class__.__name__] = componentObj
    
    # Recursively load child actors
    for child in xmlElement.find('children'):
      if child.tag == 'Actor':
        actor.children.append(cls.fromXMLElement(child, actorFactory))
    
    return actor
  
  def __init__(self, renderer, isTransient=False):
    self.renderer = renderer  # TODO make renderer a component?
    self.isTransient = isTransient  # if transient, will not be exported (in XML, etc.)
    
    # Dictionary to store all components by name
    self.components = dict()
    
    # List of child actors
    self.children = []
    
    # TODO Consider a finalize() method that copies out references to key components for fast retrieval,
    #   and possibly pre-computes some results (such as a composite transform matrix)
  
  def draw(self, transform=hm.identity()):
    # TODO draw and recurse down only when this Actor is enabled
    # TODO move to a more generic approach?
    #   e.g.:- for component in self.components: component.apply()
    #   But how do we ensure order is maintained? (Mesh must be rendered after Transform and Material have been applied)
    try:
      transform = hm.translation(transform, self.components['Transform'].translation)  # TODO make transform relative to parent, not absolute
      transform = hm.rotation(transform, self.components['Transform'].rotation[0], [1, 0, 0])
      transform = hm.rotation(transform, self.components['Transform'].rotation[1], [0, 1, 0])
      transform = hm.rotation(transform, self.components['Transform'].rotation[2], [0, 0, 1])
      transform = hm.scale(transform, self.components['Transform'].scale)
    except KeyError, AttributeError:
      # Transform component not present or incomplete/invalid
      pass  #self.renderer.setModelMatrix(transform)  # use base (parent) transform (?) - should be already set
    
    try:
      # TODO Check if this actor is visible and has a mesh, and only then render mesh
      self.renderer.setModelMatrix(transform)
      self.renderer.setDiffCol(self.components['Material'].color)
      self.components['Mesh'].render()
    except KeyError:
      # No Material and/or Mesh, silently ignore (TODO: Make this more efficient since we'll have a lot of Empty actors)
      pass
    
    for child in self.children:
      child.draw(transform)  # TODO do not draw if not enabled (e.g. an invisible actor should be enabled, but not visible)
  
  def toXMLElement(self):
    xmlElement = ET.Element(self.__class__.__name__)
    
    componentsElement = ET.SubElement(xmlElement, 'components')
    for component in self.components.itervalues():
      componentsElement.append(component.toXMLElement())
    
    childrenElement = ET.SubElement(xmlElement, 'children')
    for child in self.children:
      if not child.isTransient:  # only export non-transient children
        childrenElement.append(child.toXMLElement())
    
    return xmlElement
  
  def toString(self, indent=""):
    out = indent + "Actor: {\n"
    if self.components:
      out += indent + "  components: {\n"
      out += ",\n".join(component.toString(indent + "    ") for component in self.components.itervalues()) + "\n"
      out += indent + "  }"
      out += ",\n" if self.children else "\n"
    if self.children:
      out += indent + "  children: {\n"
      out += ",\n".join(child.toString(indent + "    ") for child in self.children if not child.isTransient) + "\n"  # only show non-transient children
      out += indent + "  }\n"
    out += indent + "}"
    
    return out
  
  def __str__(self):
    return self.toString()
