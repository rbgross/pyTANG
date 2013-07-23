import hommat as hm

class SceneNode:
  """A unit of the rendered scene that shares a common model transform."""
  
  def __init__(self):
    self.transform = hm.identity()
    self.actors = []
    self.children = []
  
  def draw():
    # TODO set common transform matrix
    for actor in self.actors:
      actor.draw()
    
    for child in self.children:
      child.draw()
    
    # TODO implement ability to show/hide actors and/or children