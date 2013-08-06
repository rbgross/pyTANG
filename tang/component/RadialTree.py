from math import pi, sin, cos, acos, atan2, hypot, copysign
import numpy as np
import hommat as hm

from Actor import Actor
from Component import Component
from Transform import Transform
from Material import Material
from Mesh import Mesh

class RadialTree(Component):
  treeRootModelFile = "TreeRoot_1.obj"
  treeEdgeModelFile = "TreeEdge_X_4_rounded.obj"
  treeEdgeLength = 4.0  # NOTE this should match the actual length in edge model file
  default_fanout = 7  # only for the first level
  default_depth = 4
  default_spread = 36.0  # max. spread, only for levels after the first one
  default_rootColor = np.float32([0.8, 0.8, 0.8])
  default_edgeColor = np.float32([0.4, 0.4, 0.4])
  default_scale = np.float32([1.0, 1.0, 1.0])
  
  @classmethod
  def fromXMLElement(cls, xmlElement, actor=None):
    fanout = xmlElement.get('fanout')
    depth = xmlElement.get('depth')
    spread = xmlElement.get('spread')
    rootColor = xmlElement.get('rootColor')
    edgeColor = xmlElement.get('edgeColor')
    scale = xmlElement.get('scale')
    return RadialTree(
      int(fanout) if fanout is not None else cls.default_fanout,
      int(depth) if depth is not None else cls.default_depth,
      float(spread) if spread is not None else cls.default_spread,
      np.fromstring(rootColor, dtype=np.float32, sep=' ') if rootColor is not None else cls.default_rootColor,
      np.fromstring(edgeColor, dtype=np.float32, sep=' ') if edgeColor is not None else cls.default_edgeColor,
      np.fromstring(scale, dtype=np.float32, sep=' ') if scale is not None else cls.default_scale,
      actor)
  
  def __init__(self, fanout, depth, spread=default_spread, rootColor=default_rootColor, edgeColor=default_edgeColor, scale=default_scale, actor=None):
    Component.__init__(self, actor)
    self.fanout = fanout
    self.depth = depth
    self.spread = spread
    self.rootColor = rootColor
    self.edgeColor = edgeColor
    self.scale = scale
    self.rootScale = self.scale  # NOTE this will also apply to children
    #self.childScale = np.float32([1.0, 1.0, 1.0])  # NOTE this will get compounded if the radial tree is a true hierarchy, better to use scale = 1 (default)
    
    # Recursively generate radial tree
    treeRoot = self.createRadialTree(self.fanout, self.depth, self.spread, isRoot=True)  # creates full hierarchy and returns root actor
    treeRoot.components['Transform'] = Transform(rotation=np.random.uniform(-pi/2, pi/2, size=3), scale=self.rootScale, actor=treeRoot)  # NOTE random rotation ensures child vectors are not always generated close to the same canonical vectors 
    treeRoot.components['Material'] = Material(color=self.rootColor, actor=treeRoot)
    treeRoot.components['Mesh'] = Mesh.getMesh(src=self.treeRootModelFile, actor=treeRoot)
    
    # Attach this hierarchy to current actor
    self.actor.children.append(treeRoot)
    
    # TODO save generated tree to XML (and load later on)
    # TODO perform a random walk down the tree to highlight a leaf node (separate method)
  
  def createRadialTree(self, fanout, depth, spread, isRoot=False):
    if depth <= 0:
      return None
    
    # * Create this node (an actor)
    treeNode = Actor(self.actor.renderer, isTransient=True)
    
    # * Attach children if depth > 1
    if depth > 1:
      if isRoot:
        # ** Pick all random directions first, to ensure a good spread, and then generate children
        childVectors = np.float32(
          [[ -1, -1, -1 ],
           [  1, -1, -1 ],
           [  1,  1, -1 ],
           [ -1,  1, -1 ],
           [ -1, -1,  1 ],
           [  1, -1,  1 ],
           [  1,  1,  1 ],
           [ -1,  1,  1 ]])  # NOTE these canonical directions are same as cube vertices!
        
        # ** Perturb child vectors, and compute unit direction vectors
        perturbation = np.random.normal(scale=1.0, size=childVectors.shape)
        #print "RadialTree.__init__(): Child:-\norig. vectors:\n", childVectors, "\nperturbation :\n", perturbation
        childVectors += perturbation
        #print "pert. vectors:\n", childVectors
        childNorms = np.apply_along_axis(lambda vec:np.linalg.norm(vec, ord=2), 1, childVectors)
        #print "child norms:\n", childNorms
        childUnits = childVectors / childNorms[:, np.newaxis]
        
        # ** Use child unit vectors one by one to create fanout first-level children
        numChildren = fanout if fanout <= childUnits.shape[0] else childUnits.shape[0]  # NOTE assert/enforce: actual fanout <= childUnits.shape[0]
        # TODO randomly pick from child unit vectors without replacement?
        for unit in childUnits[0:numChildren]:
          translation = self.treeEdgeLength * unit
          phi = np.arctan2(-unit[2], unit[0])
          theta = np.arcsin(unit[1])
          rotation = np.degrees(np.float32([ 0, phi, theta ]))  # NOTE random X-axis rotation can be added (but won't make any difference): np.random.uniform(-pi, pi)
          
          #print "RadialTree.__init__(): Child:-\nunit:", unit, "[ norm = ", np.linalg.norm(unit, ord=2), "]\ntranslation:", translation, "\nrotation:", rotation
          childNode = self.createRadialTree(np.random.random_integers(3, 4), depth - 1, spread)  # recurse down, decreasing depth
          childNode.components['Transform'] = Transform(translation=translation, rotation=rotation, actor=childNode)  # NOTE scaling will accumulate, so use scale = 1 (default)
          childNode.components['Material'] = Material(color=self.edgeColor, actor=childNode)
          childNode.components['Mesh'] = Mesh.getMesh(src=self.treeEdgeModelFile, actor=childNode)
          treeNode.children.append(childNode)
      else:
        while len(treeNode.children) < fanout:
          # ** Pick a random direction for creating a new child
          spread_rad = np.radians(spread)
          phi = np.random.uniform(-spread_rad, spread_rad)  # rotation around Y axis
          theta = np.random.uniform(-spread_rad, spread_rad)  # rotation around Z axis
          rotation = np.degrees(np.float32([ 0.0, phi, theta ]))  # this will serve as orientation for the tree edge
          
          # TODO pick a random length; scale X axis accordingly (how? flatten tree hierarchy? pick from discrete lengths and use different models accordingly?)
          
          # ** Compute relative position of new child (in cartesian coordinates) and normalized unit vector
          translation = np.float32([
            self.treeEdgeLength * cos(theta) * cos(phi),
            self.treeEdgeLength * sin(theta),
            -self.treeEdgeLength * cos(theta) * sin(phi)
          ])
          norm = np.linalg.norm(translation, ord=2)
          unit = translation / norm
          
          # TODO check closeness condition (too close to existing nodes? parent? - need absolute coordinates for that!)
          
          # ** Generate and add child (with nested tree)
          childNode = self.createRadialTree(np.random.random_integers(3, 4), depth - 1, spread)  # recurse down, decreasing depth
          childNode.components['Transform'] = Transform(translation=translation, rotation=rotation, actor=childNode)  # NOTE scaling will accumulate, so use scale = 1 (default)
          childNode.components['Material'] = Material(color=self.edgeColor, actor=childNode)
          childNode.components['Mesh'] = Mesh.getMesh(src=self.treeEdgeModelFile, actor=childNode)
          treeNode.children.append(childNode)
    
    return treeNode
  
  def toXMLElement(self):
      xmlElement = Component.toXMLElement(self)
      xmlElement.set('fanout', str(self.fanout))
      xmlElement.set('depth', str(self.depth))
      xmlElement.set('spread', str(self.spread))
      xmlElement.set('rootColor', str(self.rootColor).strip('[ ]'))
      xmlElement.set('edgeColor', str(self.edgeColor).strip('[ ]'))
      xmlElement.set('scale', str(self.scale).strip('[ ]'))
      return xmlElement
  
  def toString(self, indent=""):
      return indent + "RadialTree: {\n" + \
             indent + "  fanout: " + str(self.fanout) + ",\n" + \
             indent + "  depth: " + str(self.depth) + ",\n" + \
             indent + "  spread: " + str(self.depth) + ",\n" + \
             indent + "  rootColor: " + str(self.depth) + ",\n" + \
             indent + "  edgeColor: " + str(self.depth) + ",\n" + \
             indent + "  scale: " + str(self.scale) + "\n" + \
             indent + "}"
  
  def __str__(self):
      return self.toString()

# Register component type for automatic delegation (e.g. when inflating from XML)
Component.registerType(RadialTree)
