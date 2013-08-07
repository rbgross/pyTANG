# Python imports
import sys
from math import pi
import numpy as np
from collections import OrderedDict

# CV imports
import cv2

# Custom imports
from base import FrameProcessor
from colorfilter import HSVFilter
from input import run
#from read_mri_image import getTiles, volume_npy_file

# Flags
doRenderContours = True
doRenderMarkers = True
doRenderCube = True
doRenderFace = True
doRenderVolume = False

# Camera calibration
# TODO calibrate camera using printed pattern on cardboard, save to config file, read here
'''
camera_params = np.float64(
  [[9.7708949876746601e+02, 0., 6.2482145912532496e+02],
   [0., 9.7656102551569313e+02, 3.5368190432771917e+02],
   [0., 0., 1.]])
dist_coeffs = np.float64([-3.0379710721357184e-01, 2.1133934755379138e+00, 1.8317127842586893e-04, 4.0143088611053151e-04, -5.6225773846527973e+00])
'''
f = 750  # fx = fy = f  # focal length in pixel units
w = 640
h = 480
camera_params = np.float64(
  [[  f, 0.0, w / 2],
   [0.0,   f, h / 2],
   [0.0, 0.0,   1.0]])
dist_coeffs = np.zeros(5)  # distortion coefficients only matter for high accuracy tracking

# Bounding box/cube
cube_vertices = np.float32(
  [[ -1,  -1,  -1],
   [ 1,  -1,  -1],
   [ 1,  1,  -1],
   [ -1,  1,  -1],
   [ -1,  -1,  1],
   [ 1,  -1,  1],
   [ 1,  1,  1],
   [ -1,  1,  1]])

cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]

cube_faces = OrderedDict(
  [('front' , (0, 3, 2, 1)),
   ('back'  , (5, 6, 7, 4)),
   ('left'  , (4, 7, 3, 0)),
   ('right' , (1, 2, 6, 5)),
   ('top'   , (4, 0, 1, 5)),
   ('bottom', (3, 7, 6, 2))])

cube_scale = np.float32([10.0, 10.0, 10.0])  # TODO ensure cube is scaled correctly (check units)

cube_vertex_colors = [
  'orange', 'yellow', 'red', 'blue',
  'red', 'blue', 'orange', 'yellow' ]

colors_by_name = {
  'red': np.float32([0.8, 0.0, 0.0]),
  'green': np.float32([0.0, 0.8, 0.0]),
  'blue': np.float32([0.0, 0.0, 0.8]),
  'orange': np.float32([0.8, 0.4, 0.0]),
  'yellow': np.float32([0.8, 0.8, 0.0]) }

# Rect
#square_tag_by_vertex = ['red', 'blue', 'green', 'yellow']
#square_vertex_by_tag = { 'red': 0, 'blue': 1, 'green': 2, 'yellow': 3 }

# Cube
square_vertex_by_tag = { 'orange': 0, 'yellow': 1, 'red': 2, 'blue': 3 }
                         #'red': 4, 'blue': 5, 'orange': 6, 'yellow': 7 }  # NOTE dicts must have unique keys, so this is not a good representation
square_tag_by_vertex = { }
for tag, vertex in square_vertex_by_tag.iteritems():
  square_tag_by_vertex[vertex] = tag

# Color filters
redFilter = HSVFilter(np.array([175, 100, 75], np.uint8), np.array([5, 255, 255], np.uint8))
blueFilter = HSVFilter(np.array([100, 100, 75], np.uint8), np.array([115, 255, 255], np.uint8))
orangeFilter = HSVFilter(np.array([5, 125, 100], np.uint8), np.array([15, 255, 255], np.uint8))
#greenFilter = HSVFilter(np.array([70, 100, 75], np.uint8), np.array([90, 255, 255], np.uint8))
greenFilter = HSVFilter(np.array([60, 64, 32], np.uint8), np.array([90, 255, 255], np.uint8))  # dark green
yellowFilter = HSVFilter(np.array([16, 85, 150], np.uint8), np.array([44, 255, 255], np.uint8))
purpleFilter = HSVFilter(np.array([110, 32, 32], np.uint8), np.array([140, 255, 255], np.uint8))

class Blob:
  colorBlue = (255, 0, 0)
  colorDarkBlue = (128, 64, 64)
  
  def __init__(self, tag, area, bbox, rect):
    self.tag = tag
    self.area = area
    self.bbox = bbox
    self.rect = rect
    self.center = self.rect[0]
    self.center_int = (int(self.center[0]), int(self.center[1]))  # int type is needed for drawing functions
    self.size = self.rect[1]
    self.angle = self.rect[2]
  
  def draw(self, imageOut, drawTag=False):
    cv2.rectangle(imageOut, (self.bbox[0], self.bbox[1]), (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]), self.colorBlue, 2)
    if drawTag:
      cv2.putText(imageOut, self.tag, self.center_int, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
  
  def __str__(self):
    return "<Blob {tag} at ({center[0]:.2f}, {center[1]:.2f}), size: ({size[0]:.2f}, {size[1]:.2f})>".format(tag=self.tag, center=self.center, size=self.size)


class Trackable:
  """Abstract base class to (eventually) allow multiple independently trackable objects."""
  pass  # TODO implement this and integrate with ColorTracker


class ColorTracker(FrameProcessor):
  minBlobArea = 500
  maxBlobArea = 5000
  
  def __init__(self, options):
    FrameProcessor.__init__(self, options)
    #self.debug = False  # set to False to prevent unnecessary debug prints, esp. to record output videos
  
  def initialize(self, imageIn, timeNow):
    self.image = imageIn
    self.imageSize = (self.image.shape[1], self.image.shape[0])  # (width, height)
    self.imageOut = None
    self.active = True
    
    # * Initialize color filtering, marker structures, and 3D projection params
    #self.filterBank = dict(red=redFilter, blue=blueFilter, green=greenFilter, yellow=yellowFilter)  # Rect: RBGY
    self.filterBank = dict(red=redFilter, blue=blueFilter, orange=orangeFilter, yellow=yellowFilter)  # Cube: RBOY
    self.masks = { }
    self.morphOpenKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    self.rvec = np.zeros((3, 1), dtype=np.float32)
    self.tvec = np.zeros((3, 1), dtype=np.float32)
    
    # * Read in camera parameters
    # NOTE defined as module objects
    
    # * Initialize cube model and detection square/rect
    '''
    #scale = [375, 225, -100]  # negative Z-scale will make the model pop up and out of the screen (relative to base)
    scale = [145, 145, 145]
    #scale = [self.imageSize[0] / 4, self.imageSize[1] / 4, -(self.imageSize[1] / 8)]
    #scale = [1, 1, 1]
    shift = [self.imageSize[0] / 2 - scale[0] / 2, self.imageSize[1] / 2 - scale[1] / 2, -self.imageSize[1]]  # Z-shift doesn't really matter
    #shift = [-(scale[0] / 2), -(scale[1] / 2), -self.imageSize[1]]
    #shift = [0, 0, -self.imageSize[1]]
    self.cube_vertices = cube_vertices * scale + shift  # scaled and shifted
    '''
    
    self.cube_vertices = cube_vertices * cube_scale
    #self.cube_vertices = cube_vertices
    self.cube_edges = cube_edges
    self.base_vertices = self.cube_vertices[:4]  # first 4 vertices of cube form the base square
    self.square_vertex_by_tag = square_vertex_by_tag
    
    self.logger.debug("Camera params:\n{}".format(camera_params))
    self.logger.debug("Cube vertices:\n{}".format(self.cube_vertices))
    self.logger.debug("Cube edges:\n{}".format(self.cube_edges))
    
    # * Read in volume/point-cloud/model
    if doRenderVolume:
      model_volume_data = np.load(volume_npy_file)
      sampleStep = 500  # pick every nth point to reduce computational load
      self.model_volume_points = np.float32(model_volume_data[0::sampleStep, 0:3])  # resample, convert to float32 (needed by projectPoints)
      self.model_volume_intensities = model_volume_data[0::sampleStep, 3]
      self.logger.debug("Loaded model volume points; shape: {}, dtype: {}".format(self.model_volume_points.shape, self.model_volume_points.dtype))
      
      '''
      minX, minY, minZ = np.amin(self.model_volume_points, axis=0)
      maxX, maxY, maxZ = np.amax(self.model_volume_points, axis=0)
      model_scale = [1., 1., 1.5]
      model_shift = [self.imageSize[0] / 2 - (maxX - minX) / 2, self.imageSize[1] / 2 - (maxY - minY) / 2, self.cube_vertices[4, 2]]  # center in volume
      self.model_volume_points = self.model_volume_points * model_scale + model_shift
      self.logger.debug("Ranges:- x: [{}, {}], y: [{}, {}], z: [{}, {}]".format(minX, maxX, minY, maxY, minZ, maxZ))
      '''
      
      mins = np.amin(self.model_volume_points, axis=0)
      maxs = np.amax(self.model_volume_points, axis=0)
      # TODO compute best fit box possible, maintaining aspect-ratio, center inside cube
      model_scale = (self.cube_vertices[6] - self.cube_vertices[0]) / (maxs - mins)
      model_shift = self.cube_vertices[0]
      self.logger.debug("model_scale: {}, model_shift: {}".format(model_scale, model_shift))
      self.model_volume_points = (self.model_volume_points - mins) * model_scale + model_shift
  
  def process(self, imageIn, timeNow):
    self.image = imageIn
    #self.image = cv2.merge([cv2.equalizeHist(imageIn[:,:,0]), cv2.equalizeHist(imageIn[:,:,1]), cv2.equalizeHist(imageIn[:,:,2])])
    # TODO normalize intensity instead
    if self.gui: self.imageOut = self.image.copy()
    
    # * Initialize blobs
    self.blobs = list()
    
    # * Get HSV
    self.imageHSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
    
    # * Apply filters
    for filterName, colorFilter in self.filterBank.iteritems():
      mask = colorFilter.apply(self.imageHSV)
      # ** Smooth out mask and remove noise
      mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphOpenKernel, iterations=2)
      self.masks[filterName] = mask
      if self.gui: cv2.imshow(filterName, self.masks[filterName])
      
      # ** Detect contours in mask
      contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      if len(contours) > 0:
        #self.logger.debug("[%.2f] %d %s contour(s)" % (timeNow, len(contours), maskName))  # report contours found
        if self.gui and self.debug and doRenderContours: cv2.drawContours(self.imageOut, contours, -1, (0, 255, 255))  # draw all contours found
        
        # *** Walk through list of contours
        for contour in contours:
          #contour = contour.astype(np.int32)  # convert contours to 32-bit int for each individual contour [Pandaboard OpenCV bug workaround]
          
          # **** Filter out ones that are too small or too big
          area = cv2.contourArea(contour)
          if area < self.minBlobArea or area > self.maxBlobArea: continue
          
          # **** Create blob
          bbox = cv2.boundingRect(contour)
          rect = cv2.minAreaRect(contour)
          blob = Blob(filterName, area, bbox, rect)
          self.blobs.append(blob)
          if self.gui and self.debug and doRenderMarkers: blob.draw(self.imageOut, drawTag=True)
    
    # * Report blobs found
    if self.blobs:
      #self.logger.debug("{0} blobs found:\n{1}".format(len(self.blobs), "\n".join((str(blob) for blob in self.blobs))))  # verbose
      self.logger.debug("{0} blobs found: {1}".format(len(self.blobs), ", ".join((blob.tag for blob in self.blobs))))  # brief
    else:
      return True, self.imageOut  # nothing more to do, bail out
    
    '''
    # [Method 1. Try to match *base* face (i.e. a single face)]
    # * Map blob centers to base vertex points
    self.base_points = [None] * len(self.base_vertices)
    for blob in self.blobs:
      self.base_points[self.square_vertex_by_tag[blob.tag]] = blob.center  # NOTE last blob of a particular tag overwrites any previous blobs with same tag
    
    foundBasePoints = True
    for point in self.base_points:
      if point is None:
        foundBasePoints = False
        self.logger.debug("Warning: Base point not detected")
        return True, self.imageOut  # skip
        #break  # keep using last transform; TODO set rvec, tvec to None if tracking is lost for too long
    
    # * If all base points are found, compute 3D projection/transform (as separate rotation and translation vectors: rvec, tvec)
    if foundBasePoints:
      self.base_points = np.float32(self.base_points)
      #self.logger.debug("Base points:\n{}".format(self.base_points))
      
      retval, self.rvec, self.tvec = cv2.solvePnP(self.base_vertices, self.base_points, camera_params, dist_coeffs)
      self.logger.debug("\nretval: {}\nrvec: {}\ntvec: {}".format(retval, self.rvec, self.tvec))
    '''
    
    # [Method 2. Try to match *any* face that might be visible]
    # * For each (named) cube face
    for name, face in cube_faces.iteritems():
      # ** Obtain face vertices, vertex colors, and vertex color to index mapping (NOTE color == tag)
      face_vertices = np.float32([self.cube_vertices[vertex_idx] for vertex_idx in face])
      face_vertex_colors = [cube_vertex_colors[vertex_idx] for vertex_idx in face]
      #print "face:", face, "\nface_vertices:", face_vertices, "\nface_vertex_colors:", face_vertex_colors
      face_vertex_idx_by_color = OrderedDict(zip(face_vertex_colors, range(len(face_vertex_colors))))
      #print "face_vertex_idx_by_color:", face_vertex_idx_by_color
      
      # ** Map blob centers to face vertex points
      face_points = [None] * len(face)
      for blob in self.blobs:
        face_points[face_vertex_idx_by_color[blob.tag]] = blob.center  # NOTE last blob of a particular tag overwrites any previous blobs with same tag
      
      # ** Check if all face points have been found
      isValidFace = True  # TODO split these verification steps into different methods (or one?) and avoid if-thens
      for point in face_points:
        if point is None:
          isValidFace = False
          self.logger.debug("Warning: Face point not detected")
          break  # this face is incomplete, maybe some other face will match
      
      # ** If all face points are found, verify that they satisfy known topographical structure (order)
      if isValidFace:
        face_points = np.float32(face_points)
        #print "face_points:\n{}".format(face_points)
        
        # NOTE Topographical structure is inherent in the order of face points (should be counter-clockwise)
        # *** Compute centroid of face points
        face_centroid = np.mean(face_points, axis=0)
        #print "face_centroid: {}".format(face_centroid)
        
        # *** [2D] Compute heading angles to each point assuming face normal is pointing outward through the screen
        heading_vecs = face_points - face_centroid
        headings = np.float32([ np.arctan2(vec[1], vec[0]) for vec in heading_vecs ])
        
        # *** [3D] Compute face plane normal and heading angles to each point around the normal (TODO)
        #face_normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[3] - face_vertices[0])
        #print "face_normal: {}".format(face_normal)
        
        # *** These headings should be in decreasing order (since they are counter-clockwise, and Y-axis is downwards)
        heading_diffs = headings - np.roll(headings, 1)
        heading_diffs = ((heading_diffs + pi) % (2 * pi)) - pi  # ensures angle wraparound is handled correctly
        #print "heading_diffs: {}".format(heading_diffs)
        if np.any(heading_diffs > 0):
          isValidFace = False  # if any heading angle difference is positive, skip this face
        
      # ** Check if face points form a convex quadrilateral facing us
      if isValidFace:
        #print "Face: {}".format(name)
        #print "face_points:\n{}".format(face_points)
        edge_vectors = face_points - np.roll(face_points, 1, axis=0)  # vectors between adjacent vertices (i.e. along edges)
        #print "edge_vectors:\n{}".format(edge_vectors)
        cross_prods = np.cross(edge_vectors, np.roll(edge_vectors, 1, axis=0))  # cross products of consecutive edge vectors
        #print "cross_prods: {}".format(cross_prods)
        if np.any(cross_prods < 0):
          isValidFace = False  # if any cross product is negative, then quad is either non-convex or back-facing
      
      # ** Check if opposite heading vector lengths and angles are almost equal (i.e. the quad is almost a rhombus)
      if isValidFace:
        heading_lens = np.float32([np.hypot(vec[0], vec[1]) for vec in heading_vecs])
        #print "heading_lens: {}".format(heading_lens)
        isValidFace = abs(heading_lens[0] - heading_lens[2]) / max(heading_lens[0], heading_lens[2]) < 0.5 \
                  and abs(heading_lens[1] - heading_lens[3]) / max(heading_lens[1], heading_lens[3]) < 0.5 \
                  and abs(heading_diffs[0] - heading_diffs[2]) / max(heading_diffs[0], heading_diffs[2]) < 0.25 \
                  and abs(heading_diffs[1] - heading_diffs[3]) / max(heading_diffs[1], heading_diffs[3]) < 0.25
      
      # ** Compute 3D projection/transform if a valid face is found (as separate rotation and translation vectors: rvec, tvec)
      if isValidFace:
        self.logger.debug("Valid face: {}".format(name))
        if self.gui and doRenderFace:
          face_centroid_int = (int(face_centroid[0]), int(face_centroid[1]))
          cv2.circle(self.imageOut, face_centroid_int, 10, (0, 128, 0), -1)
          for point, heading in zip(face_points, headings):
            cv2.line(self.imageOut, face_centroid_int, (int(point[0]), int(point[1])), (0, 128, 0), 2)
            #label_pos = (face_centroid_int[0] + int(100 * np.cos(heading)), face_centroid_int[1] + int(100 * np.sin(heading)))
            #cv2.putText(self.imageOut, "{:.2f}".format(heading * 180 / pi), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
          cv2.putText(self.imageOut, name, face_centroid_int, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        retval, self.rvec, self.tvec = cv2.solvePnP(face_vertices, face_points, camera_params, dist_coeffs)
        self.logger.debug("\nretval: {}\nrvec: {}\ntvec: {}".format(retval, self.rvec, self.tvec))
        break  # use the first complete face that is found, and skip the rest
        # TODO compute multiple transforms and pick best one (closest to last transform, cube center within bounds)
    
    # * Check if we have a valid transform
    if self.rvec is None or self.tvec is None:
      return True, self.imageOut  # skip rendering
    
    # * Project a cube overlayed on top of video stream
    if self.gui and doRenderCube:
      cube_points, jacobian = cv2.projectPoints(self.cube_vertices, self.rvec, self.tvec, camera_params, dist_coeffs)
      cube_points = cube_points.reshape(-1, 2)  # remove nesting
      #self.logger.debug("Projected cube points:\n{}".format(cube_points))
      for u, v in self.cube_edges:
        if u < len(cube_points) and v < len(cube_points) and cube_points[u] is not None and cube_points[v] is not None:  # sanity check
          cv2.line(self.imageOut, (int(cube_points[u][0]), int(cube_points[u][1])), (int(cube_points[v][0]), int(cube_points[v][1])), (255, 255, 0), 2)
    
    # TODO Project a visualization/model overlayed on top of video stream
    if self.gui and doRenderVolume:
      volume_points, jacobian = cv2.projectPoints(self.model_volume_points, self.rvec, self.tvec, camera_params, dist_coeffs)
      volume_points = volume_points.reshape(-1, 2)  # remove nesting
      for point, intensity in zip(volume_points, self.model_volume_intensities):
        if 0 <= point[0] < self.imageSize[0] and 0 <= point[1] < self.imageSize[1]:
          self.imageOut[point[1], point[0]] = (intensity, intensity, intensity)
    
    return True, self.imageOut

if __name__ == "__main__":
  options = { 'gui': True, 'debug': ('--debug' in sys.argv) }
  run(ColorTracker(options=options), gui=options['gui'], debug=options['debug'])
