from math import copysign, acos, atan2, hypot
import numpy as np

from Actor import Actor
from Component import Component
from Transform import Transform
from Material import Material
from Mesh import Mesh
from vision.colortracking import Trackable, ColorMarker, cube_vertices, cube_edges, cube_scale, cube_vertex_colors, colors_by_name

class Cube(Component, Trackable):
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    scale = xmlElement.get('scale')
    return Cube(
      np.fromstring(scale, dtype=np.float32, sep=' ') if scale is not None else cube_scale,
      actor)
  
  def __init__(self, scale=cube_scale, actor=None):
    Component.__init__(self, actor)
    Trackable.__init__(self)
    self.scale = scale
    
    # Scale vertices of base cube, specify edges, and initialize list of markers
    self.vertices = cube_vertices * self.scale
    self.vertex_colors = cube_vertex_colors
    self.vertex_scale = 0.3 * self.scale  # NOTE for rendering only, depends on 3D model
    
    self.edges = cube_edges
    self.edge_scale = 0.1 * self.scale  # NOTE for rendering only, depends on 3D model
    self.edge_color = np.float32([0.8, 0.7, 0.5])  # NOTE for rendering only
    # TODO make some of these parameters come from XML
    
    # NOTE Mark generated child actors (corners and edges) as transient, to prevent them from being exported in XML
    
    # Add spheres at cube corners (vertices) with appropriate color; also add color markers
    for vertex, colorName in zip(self.vertices, self.vertex_colors):
      vertexActor = Actor(self.actor.renderer, isTransient=True)
      vertexActor.components['Transform'] = Transform(translation=vertex, scale=self.vertex_scale, actor=vertexActor)
      vertexActor.components['Material'] = Material(color=colors_by_name[colorName], actor=vertexActor)
      vertexActor.components['Mesh'] = Mesh.getMesh(src="SmallSphere.obj", actor=vertexActor)
      self.actor.children.append(vertexActor)
      marker = ColorMarker(self, colorName)
      marker.worldPos = vertex
      self.markers.append(marker)
    
    # Add edges
    for u, v in self.edges:
      if u < len(self.vertices) and v < len(self.vertices) and self.vertices[u] is not None and self.vertices[v] is not None:  # sanity check
        midPoint = (self.vertices[u] + self.vertices[v]) / 2.0
        diff = self.vertices[v] - self.vertices[u]
        mag = np.linalg.norm(diff, ord=2)
        xy_mag = hypot(diff[0], diff[1])
        #zx_mag = hypot(diff[2], diff[0])
        rotation = np.degrees(np.float32([atan2(diff[1], diff[0]), acos(diff[1] / mag), 0])) if (mag != 0 and xy_mag != 0) else np.float32([0.0, 0.0, 0.0])
        #print "u: ", self.vertices[u], ", v: ", self.vertices[v], ", v-u: ", diff, ", mag: ", mag, ", rot:", rotation
        edgeActor = Actor(self.actor.renderer, isTransient=True)
        edgeActor.components['Transform'] = Transform(translation=midPoint, rotation=rotation, scale=self.edge_scale, actor=edgeActor)
        edgeActor.components['Material'] = Material(color=self.edge_color, actor=edgeActor)
        edgeActor.components['Mesh'] = Mesh.getMesh(src="CubeEdge.obj", actor=edgeActor)  # TODO fix Z-fighting issue and use CubeEdge_cylinder.obj
        self.actor.children.append(edgeActor)
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      xmlElement.set('scale', str(self.scale).strip('[ ]'))
      return xmlElement
  
  def toString(self, indent=""):
      return indent + "Cube: { scale: " + str(self.scale) + " }"
  
  def __str__(self):
      return self.toString()

# Register component type for automatic delegation (e.g. when inflating from XML)
Component.registerType(Cube)
