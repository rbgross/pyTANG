# Python imports
import sys
from math import pi, hypot
import numpy as np
from collections import OrderedDict
from Queue import PriorityQueue
from itertools import permutations, combinations
import logging

# CV imports
import cv2
import cv2.cv as cv

# Custom imports
from base import FrameProcessor
from colorfilter import HSVFilter
from input import run
#from read_mri_image import getTiles, volume_npy_file

# Flags
doRenderContours = True
doRenderBlobs = False
doRenderMarkers = True
doRenderCube = False
doRenderFace = True
doRenderVolume = False
doSmoothPose = True

# Global constants
two_pi = 2 * pi

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
  [[-1, -1, -1],
   [ 1, -1, -1],
   [ 1,  1, -1],
   [-1,  1, -1],
   [-1, -1,  1],
   [ 1, -1,  1],
   [ 1,  1,  1],
   [-1,  1,  1]])

cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]

cube_adj = np.zeros(shape=(len(cube_vertices), len(cube_vertices)), dtype=np.uint8)
for u, v in cube_edges:
  cube_adj[u, v] = 1
  cube_adj[v, u] = 1

cube_faces = OrderedDict(
  [('front' , (0, 3, 2, 1)),
   ('back'  , (5, 6, 7, 4)),
   ('left'  , (4, 7, 3, 0)),
   ('right' , (1, 2, 6, 5)),
   ('top'   , (4, 0, 1, 5)),
   ('bottom', (3, 7, 6, 2))])

cube_scale = np.float32([10.0, 10.0, 10.0])  # TODO ensure cube is scaled correctly (check units)

'''
# Color assignment #1: red, orange, yellow, blue
cube_vertex_colors = [
  'orange', 'yellow', 'red', 'blue',
  'red', 'blue', 'orange', 'yellow' ]
'''

# Color assignment #2: red, orange, green, blue
cube_vertex_colors = [
  'orange', 'green', 'red', 'blue',
  'red', 'blue', 'orange', 'green' ]

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
redFilter = HSVFilter(np.array([175, 115, 64], np.uint8), np.array([5, 255, 255], np.uint8))
blueFilter = HSVFilter(np.array([100, 64, 75], np.uint8), np.array([115, 255, 255], np.uint8))
#orangeFilter = HSVFilter(np.array([5, 125, 100], np.uint8), np.array([15, 255, 255], np.uint8))  # strict orange
orangeFilter = HSVFilter(np.array([5, 120, 125], np.uint8), np.array([20, 255, 255], np.uint8))  # orange with a little yellow
greenFilter = HSVFilter(np.array([50, 64, 32], np.uint8), np.array([80, 255, 255], np.uint8))  # wide range
#greenFilter = HSVFilter(np.array([70, 64, 32], np.uint8), np.array([90, 255, 255], np.uint8))  # dark green
yellowFilter = HSVFilter(np.array([20, 85, 150], np.uint8), np.array([44, 255, 255], np.uint8))
purpleFilter = HSVFilter(np.array([110, 115, 64], np.uint8), np.array([140, 255, 255], np.uint8))
# TODO multiple color filters per color to specify non-convex boundaries?

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


class Marker:
  """Base class for all markers with a real-world (3D) position and an image (2D) position."""
  
  def __init__(self, parent):
    self.parent = parent
    self.lastImagePos = self.imagePos = None  # np.zeros(2, dtype=np.float32)
    self.worldPos = None  # np.zeros(3, dtype=np.float32)
    self.active = False
  
  def updateImagePos(self, imagePos):
    self.lastImagePos = self.imagePos
    self.imagePos = imagePos
  
  def draw(self, imageOut, drawTag=False):
    if self.imagePos is not None:
      imagePos_int = (int(self.imagePos[0]), int(self.imagePos[1]))
      if self.lastImagePos is not None:
        lastImagePos_int = (int(self.lastImagePos[0]), int(self.lastImagePos[1]))
        cv2.line(imageOut, lastImagePos_int, imagePos_int, (128, 255, 0), 2)
        #cv2.circle(imageOut, lastImagePos_int, 2, (128, 0, 0), -1)
      cv2.circle(imageOut, imagePos_int, 2, (255, 0, 0), -1)
      if drawTag:
        cv2.putText(imageOut, self.tag, (imagePos_int[0] + 15, imagePos_int[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


class ColorMarker(Marker):
  """Markers with a consistent color that can be used to detect them."""
  
  def __init__(self, parent, tag=None):
    Marker.__init__(self, parent)
    self.tag = tag
    # TODO Register instances upon creation (or explicitly?) to enable tracking as a bunch


class PlanarMarker(Marker):
  """AR-Tag style planar marker."""
  # TODO Move this to its own module ("like planetracking")
  
  def __init__(self, parent, normal=np.float32([0.0, -1.0, 0.0])):  # default normal: pointing up
    Marker.__init__(self, parent)
    self.normal = normal


class Trackable:
  """Base class to allow multiple independently trackable objects."""
  
  def __init__(self):
    self.rvec = np.zeros((3, 1), dtype=np.float32)
    self.tvec = np.zeros((3, 1), dtype=np.float32)
    self.visible = False  # TODO use a better state-machine scheme
    self.markers = list()


class ColorMarkerTracker(FrameProcessor):
  """Detects ColorMarker objects in an image, using spatial continuity and proximity heuristics."""
  
  minBlobArea = 500
  maxBlobArea = 5000
  
  def __init__(self, options):
    FrameProcessor.__init__(self, options)
    #self.debug = False  # set to False to prevent unnecessary debug prints, esp. to record output videos
    
    # * Create list of trackables and markers
    self.trackables = []
    self.markers = []
  
  def addMarkersFromTrackable(self, trackable):
    self.trackables.append(trackable)
    self.logger.debug("Trackable object has {} markers: {}".format(len(trackable.markers), ", ".join(marker.__class__.__name__ for marker in trackable.markers)))
    self.markers.extend([marker for marker in trackable.markers if isinstance(marker, ColorMarker)])
    self.logger.debug("Markers added; current total: {}".format(len(self.markers)))
  
  def initialize(self, imageIn, timeNow):
    self.image = imageIn
    self.imageSize = (self.image.shape[1], self.image.shape[0])  # (width, height)
    self.imageCenter = (self.imageSize[0] / 2, self.imageSize[1] / 2)
    self.imageOut = None
    self.active = True
    
    # * Initialize color filtering structures (note: cube_vertex_colors need t be changed as well)
    #self.filterBank = dict(red=redFilter, blue=blueFilter, orange=orangeFilter, yellow=yellowFilter)  # Cube: RBOY
    self.filterBank = dict(red=redFilter, blue=blueFilter, green=greenFilter, orange=orangeFilter)  # Rect: RGBO
    #self.filterBank = dict(red=redFilter, blue=blueFilter, green=greenFilter, purple=purpleFilter)  # Rect: RGBP
    self.masks = { }
    self.morphOpenKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  
  def process(self, imageIn, timeNow):
    self.imageIn = imageIn  # keep a reference to the original image
    self.image = self.imageIn
    if self.gui: self.imageOut = self.image.copy()
    self.image = cv2.blur(self.image, (5, 5))
    #self.image = cv2.merge([cv2.equalizeHist(imageIn[:,:,0]), cv2.equalizeHist(imageIn[:,:,1]), cv2.equalizeHist(imageIn[:,:,2])])
    # TODO normalize intensity instead
    
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
          if self.gui and doRenderBlobs: blob.draw(self.imageOut, drawTag=True)
    
    # * Process blobs to find matching markers
    if self.blobs:
      self.blobs = sorted(self.blobs, key=lambda blob: blob.area, reverse=True)  # sort by decreasing size
      # ** Report blobs found
      #self.logger.debug("{0} blobs found:\n{1}".format(len(self.blobs), "\n".join((str(blob) for blob in self.blobs))))  # verbose
      self.logger.debug("{0} blobs found: {1}".format(len(self.blobs), ", ".join((blob.tag for blob in self.blobs))))  # brief
      
      # ** Match markers with blobs
      self.matchMarkersWithBlobs(self.markers, self.blobs, 100*100)
      
      # ** Report active markers
      self.logger.debug("{} active markers: {}".format(len(self.markers), ", ".join((marker.tag for marker in self.markers if marker.active))))
      if self.gui and doRenderMarkers:
        for marker in (marker for marker in self.markers if marker.active):
          marker.draw(self.imageOut, drawTag=True)
    
    return self.imageOut
  
  def matchMarkersWithBlobs(self, markers, blobs, maxDistSq=np.inf):
    '''
    # [Matching method 1] Naive method
    for marker in self.markers:
      if marker.active:
        # *** For active markers, use their last position as a search constraint/hint
        #bestBlob = self.getNearestBlob(marker.tag, marker.imagePos, 100)  # specify a max dist to rule out distant false-positives
        #bestBlobs = self.getBlobsByDistance(marker.tag, marker.imagePos)  # returns list of Blob objects
        bestBlobs = self.getBlobsByDist(marker.tag, marker.imagePos, 100)  # returns (<dist>, <Blob>) pairs
        bestBlobs = [blob for dist, blob in bestBlobs]
        if bestBlobs:  #if bestBlob is not None:
          self.logger.debug("Marker *{}*: {} matching blob(s)".format(marker.tag, len(bestBlobs)))
          marker.blobs = bestBlobs
          
          bestBlob = bestBlobs[0]
          #bestDist, bestBlob = bestBlobs[0]  # NOTE getBlobsByDist() returns (dist, blob tuple)
          marker.updateImagePos(bestBlob.center)
          self.blobs.remove(bestBlob)  # only one marker per blob
        else:
          marker.updateImagePos(None)
          marker.active = False  # TODO only mark inactive when blob hasn't been seen for a while
      else:
        # *** For inactive markers, try to find the best possible match (TODO retain multiple close matches, and disambiguate later based on whichever combination makes the most *sense*)
        bestBlob = self.getBlob(marker.tag)
        if bestBlob is not None:
          marker.blobs = [bestBlob]
          marker.updateImagePos(bestBlob.center)
          #self.blobs.remove(bestBlob)  # only one marker per blob
          marker.active = True
    '''
    
    # [Matching method 2] Based on Nearest-neighbor linker by Jean-Yves Tinevez
    # Source: http://www.mathworks.com/matlabcentral/fileexchange/33772-nearest-neighbor-linker
    
    # * Initialize distance matrix
    D = np.empty(shape=(len(markers), len(blobs)), dtype=np.float32)
    D.fill(np.inf)
    
    # * Build up distance matrix, only for valid matches (i.e. marker and blob must have same tag)
    # TODO use richer features, such as color histogram of the image region
    for i in xrange(len(markers)):
      #self.logger.debug("[Distance matrix loop] i: {}".format(i))
      markers[i].active = False
      for j in xrange(len(blobs)):
        #self.logger.debug("[Distance matrix loop]   j: {}, blob.center: {}, imageCenter: {}".format(j, blobs[j].center, self.imageCenter))
        if not markers[i].tag == blobs[j].tag: continue
        if markers[i].active:
          D[i, j] = (markers[i].imagePos[0] - blobs[j].center[0]) * (markers[i].imagePos[0] - blobs[j].center[0]) + (markers[i].imagePos[1] - blobs[j].center[1]) * (markers[i].imagePos[1] - blobs[j].center[1])
          if D[i, j] > maxDistSq: D[i, j] = np.inf  # distances beyond maxDist will not be considered
        else:
          D[i, j] = ((self.imageCenter[0] - blobs[j].center[0]) * (self.imageCenter[0] - blobs[j].center[0]) + (self.imageCenter[1] - blobs[j].center[1]) * (self.imageCenter[1] - blobs[j].center[1])) * 2  # double the distance: hack to prevent inactive markers from stealing blobs
    
    #self.logger.debug("Computed distance matrix D:\n{}".format(D))
    
    # * Iteratively find matchings for markers
    markersMatched = 0
    blobsProcessed = 0
    while markersMatched <= len(markers) and blobsProcessed <= len(blobs):
      minIdx = D.argmin()
      i, j = minIdx / len(blobs), minIdx % len(blobs)
      if np.isinf(D[i, j]): break  # no more non-inf. distance pairs left
      if not markers[i].active:
        markers[i].active = True
        #self.logger.debug("Min. D[{}, {}] = {} (pair: {} / {})".format(i, j, D[i, j], markers[i].tag, blobs[j].tag))
        markers[i].imagePos = blobs[j].center
        markers[i].active = True
        #D[i, j] = np.inf  # mark distance as infinity so that it is not chosen again
        D[i, :] = np.inf  # mark whole row as infinity (i.e. remove marker from further calculations)
        D[:, j] = np.inf  # mark whole column as infinity (i.e. remove blob from further calculations)
        #self.logger.debug("D:\n{}".format(D))
        markersMatched += 1
        blobsProcessed += 1
  
  def getBlobs(self, tag=None):
    """Return a generator/list of blobs that match given tag (or all, if not given)."""
    if tag is not None:
      return (blob for blob in self.blobs if blob.tag == tag)
    else:
      self.blobs
  
  def getBlob(self, tag=None):
    """Return a single blob that matches tag (if given)."""
    for blob in self.getBlobs(tag):
      return blob  # return the first one that matches tag
    return None
  
  def getBlobsByDistance(self, tag=None, point=None):
    if point is None: point = self.imageCenter
    nearestBlobs = sorted((blob for blob in self.getBlobs(tag)), key=lambda blob: hypot(blob.center[0] - point[0], blob.center[1] - point[1]))
    return nearestBlobs
  
  def getBlobsByDist(self, tag=None, point=None, maxDist=np.inf):
    if point is None: point = self.imageCenter
    nearestBlobs = PriorityQueue()
    for blob in self.getBlobs(tag):
      dist = hypot(blob.center[0] - point[0], blob.center[1] - point[1])
      if dist <= maxDist:
        nearestBlobs.put((dist, blob))  # insert into priority queue
    return nearestBlobs.queue  # return internal list, not to be modified
  
  def getNearestBlob(self, tag=None, point=None, maxDist=np.inf):
    if point is None: point = self.imageCenter
    minDist = maxDist
    nearestBlob = None
    for blob in self.getBlobs(tag):
      dist = hypot(blob.center[0] - point[0], blob.center[1] - point[1])
      if dist < minDist:
        minDist = dist
        nearestBlob = blob
    return nearestBlob
    
    '''
    # Alt. implementation (might not be as efficient)
    for blob in getBlobsByDistance(tag, point):
      if hypot(blob.center[0] - point[0], blob.center[1] - point[1]) < maxDist:
        return blob
      break
    return None
    '''


class CubeTracker(ColorMarkerTracker):
  inactiveThreshold = 1.2  # how many seconds to allow the cube to remain invisible before marking it inactive
  cube_origin = np.float32([[0.0], [1.72], [60.92]])  # expected cube origin
  tvec_maxdiff_origin = 40.0  # maximum distance from origin for a valid detection
  tvec_maxdiff_last = 5.0  # maximum distance from last known position for a valid detection
  rvec_maxdiff_last = pi  # maximum combined (L2-norm) angle difference from last known pose
  max_line_blob_distance_sq = 50 * 50  # squared distance between line end points and blob centers for a match
  num_smooth_samples = 5  # number of sample transformations to use for smoothing (to reduce jitter)
  
  def __init__(self, options):
    ColorMarkerTracker.__init__(self, options)
  
  def initialize(self, imageIn, timeNow):
    ColorMarkerTracker.initialize(self, imageIn, timeNow)
    
    # * Initialize members needed for cube-tracking
    #self.cubeEdgeFilter = HSVFilter(np.array([0, 40, 100], np.uint8), np.array([179, 100, 255], np.uint8))  # bare wooden sticks
    self.cubeEdgeFilter = HSVFilter(np.array([0, 0, 200], np.uint8), np.array([179, 30, 255], np.uint8))  # white sticks
    
    # * Initialize 3D projection params
    self.rvecRaw = np.zeros((3, 1), dtype=np.float32)
    self.tvecRaw = np.zeros((3, 1), dtype=np.float32)
    self.active = False
    self.lastSeen = timeNow
    self.smoothReset()
    
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
    
    self.cube_vertices = cube_vertices * cube_scale
    #self.cube_vertices = cube_vertices
    self.cube_edges = cube_edges
    self.base_vertices = self.cube_vertices[:4]  # first 4 vertices of cube form the base square
    self.square_vertex_by_tag = square_vertex_by_tag
    
    self.logger.debug("Camera params:\n{}".format(camera_params))
    self.logger.debug("Cube vertices:\n{}".format(self.cube_vertices))
    self.logger.debug("Cube edges:\n{}".format(self.cube_edges))
    '''
    
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
    ColorMarkerTracker.process(self, imageIn, timeNow)
    #if not self.blobs:
    #  return self.imageOut  # nothing more to do, bail out
    
    # * For each trackable
    for trackable in self.trackables:
      activeMarkers = [marker for marker in trackable.markers if marker.active]
      self.logger.debug("Tracking a {}: #active markers = {}".format(trackable.__class__.__name__, len(activeMarkers)))
      trackable.visible = False  # assume not visible, and proceed
      
      # ** If this trackable is a cube, use specialized tracking method
      if trackable.__class__.__name__ == 'Cube':
        found = False
        # *** Try finding the cube using a combination of observed blobs and line segments (connecting edges)
        #if not found and len(self.blobs) >= 4: found = self.trackCubeWithEdges(trackable, activeMarkers)  # NOTE does not use markers, uses blobs directly instead
        
        if len(activeMarkers) < 4:
          continue  # not enough markers for this trackable
        
        # *** Try finding the cube using faces (most stable, but requires all 4 vertices of a face to be seen)
        if not found: found = self.trackCubeMultiface2(trackable, activeMarkers)  # NOTE uses activeMarkers
        
        # *** Try matching an arbitrary set of markers (at least 4)
        if not found: found = self.trackVerticesRansac(trackable, activeMarkers)  # NOTE uses activeMarkers
        
        # *** If cube is visible, copy rotation and translation vectors for global use
        if trackable.visible:  # alt.: found
          if self.updateTransform(trackable.rvec, trackable.tvec):  # update self transform, filtering and rejecting outliers
            self.active = True
            self.lastSeen = timeNow
          else:
            if timeNow - self.lastSeen > self.inactiveThreshold:
              self.active = False
          # NOTE we can simply use the cube's vectors when projecting data/models inside it by parenting them to it
          
          # **** Project a cube overlayed on top of video stream
          if self.gui and doRenderCube:
            cube_points, jacobian = cv2.projectPoints(trackable.vertices, trackable.rvec, trackable.tvec, camera_params, dist_coeffs)
            cube_points = cube_points.reshape(-1, 2)  # remove nesting
            #self.logger.debug("Projected cube points:\n{}".format(cube_points))
            for u, v in trackable.edges:
              if u < len(cube_points) and v < len(cube_points) and cube_points[u] is not None and cube_points[v] is not None:  # sanity check
                cv2.line(self.imageOut, (int(cube_points[u][0]), int(cube_points[u][1])), (int(cube_points[v][0]), int(cube_points[v][1])), (255, 255, 0), 2)
        else:
          if timeNow - self.lastSeen > self.inactiveThreshold:
            self.active = False
      else:
        self.trackVerticesRansac(trackable, activeMarkers)
    
    # * If we have a valid transform, project a visualization/model overlayed on top of video stream
    if self.rvecRaw is not None and self.tvecRaw is not None:
      self.logger.debug("Transform [final]:-\nrvec:\n{}\ntvec:\n{}".format(self.rvecRaw, self.tvecRaw))
      if self.gui and doRenderVolume:
        volume_points, jacobian = cv2.projectPoints(self.model_volume_points, self.rvecRaw, self.tvecRaw, camera_params, dist_coeffs)
        volume_points = volume_points.reshape(-1, 2)  # remove nesting
        for point, intensity in zip(volume_points, self.model_volume_intensities):
          if 0 <= point[0] < self.imageSize[0] and 0 <= point[1] < self.imageSize[1]:
            self.imageOut[point[1], point[0]] = (intensity, intensity, intensity)
    
    return self.imageOut
  
  def trackCubeFace(self, trackable, activeMarkers, faceIdx=0):
    # TODO clean up and refactor to use trackable.* instead of self.*
    # [Tracking method 1] Try to match a single cube face (indicated by faceIdx)
    # * Map blob centers to base vertex points
    self.base_points = [None] * len(self.base_vertices)
    for blob in self.blobs:
      self.base_points[self.square_vertex_by_tag[blob.tag]] = blob.center  # NOTE last blob of a particular tag overwrites any previous blobs with same tag
    
    foundBasePoints = True
    for point in self.base_points:
      if point is None:
        foundBasePoints = False
        self.logger.debug("Warning: Base point not detected")
        return self.imageOut  # skip
        #break  # keep using last transform; TODO set rvec, tvec to None if tracking is lost for too long
    
    # * If all base points are found, compute 3D projection/transform (as separate rotation and translation vectors: rvec, tvec)
    if foundBasePoints:
      self.base_points = np.float32(self.base_points)
      #self.logger.debug("Base points:\n{}".format(self.base_points))
      
      retval, trackable.rvec, trackable.tvec = cv2.solvePnP(self.base_vertices, self.base_points, camera_params, dist_coeffs)
      self.logger.debug("\nretval: {}\nrvec: {}\ntvec: {}".format(retval, trackable.rvec, trackable.tvec))
  
  def trackCubeMultiface(self, trackable, activeMarkers):
    # [Tracking method 2-1] Try to match *any* face of the cube that might be visible
    # * For each (named) cube face
    for name, face in cube_faces.iteritems():
      # ** Obtain face vertices, vertex colors, and vertex color to index mapping (NOTE color == tag)
      face_vertices = np.float32([trackable.vertices[vertex_idx] for vertex_idx in face])
      face_markers = (trackable.markers[vertex_idx] for vertex_idx in face)
      face_vertex_colors = [cube_vertex_colors[vertex_idx] for vertex_idx in face]
      #print "face:", face, "\nface_vertices:", face_vertices, "\nface_vertex_colors:", face_vertex_colors
      face_vertex_idx_by_color = OrderedDict(zip(face_vertex_colors, range(len(face_vertex_colors))))
      #print "face_vertex_idx_by_color:", face_vertex_idx_by_color
      
      # ** Map blob centers to face vertex points, keeping a list of all candidate blobs for each vertex point
      face_points = [None] * len(face)
      face_vertex_markers = [[]] * len(face)
      for marker in face_markers:
        if not marker.active: continue
        face_vertex_markers[face_vertex_idx_by_color[marker.tag]].append(marker)
        face_points[face_vertex_idx_by_color[marker.tag]] = marker.imagePos  # NOTE last marker of a particular tag overwrites any previous markers with same tag; to get other marker positions, keep popping from face_vertex_markers and copying in imagePos
      
      # ** Check if all face points have been found
      isFaceComplete = np.all([point is not None for point in face_points])
      if not isFaceComplete:
        self.logger.debug("Incomplete face: {}".format(name))
        continue  # this face is incomplete, maybe some other face will match
      
      # ** Ensure this is a valid face using known topology
      face_points = np.float32(face_points)
      #print "face_points:\n{}".format(face_points)
      
      # ** Iterate over all candidate face-marker mappings
      # TODO use a different scheme to generate combinations; this one is flawed!
      haveCandidate = True
      while haveCandidate:
        self.logger.debug("Candidate face: {}".format(name))
        # *** If a valid face is found, compute 3D projection/transform
        if self.isCubeFaceValid(name, face_points):
          self.logger.debug("Valid face: {}".format(name))
          
          retval, trackable.rvec, trackable.tvec = cv2.solvePnP(face_vertices, face_points, camera_params, dist_coeffs)
          #self.logger.debug("Transform:-\nretval: {}\nrvec:\n{}\ntvec:\n{}".format(retval, trackable.rvec, trackable.tvec))
          trackable.visible = True
          return True  # use the first complete face that is found, and skip the rest
          # TODO compute multiple transforms and pick best one (closest to last transform, cube center within bounds)
        
        # *** Else try a different vertex-blob combination, if available
        haveCandidate = False
        for i in xrange(len(face_points)):
          if len(face_vertex_markers[i]) > 1:  # if there are more than one possible markers still left
            face_vertex_markers[i].pop()  # first, remove the current one
            face_points[i] = face_vertex_markers[i][0].imagePos  # then copy in next's position
            haveCandidate = True
            break
    
    return False
  
  def trackCubeMultiface2(self, trackable, activeMarkers):
    # [Tracking method 2-2] Try to find *all* faces that are visible and use the "best"
    
    # * Iterate over all 4-element (ordered) subsets of activeMarkers
    face_marker_seqs = []  # list of (name, face, marker_seq, face_points) tuples
    marker_sets = list(combinations(activeMarkers, 4))
    #self.logger.debug("len(marker_sets) = {}".format(len(marker_sets)))
    #total_marker_seqs = 0
    #valid_marker_seqs = 0
    for marker_set in marker_sets:
      if marker_set[0].tag == marker_set[1].tag or marker_set[0].tag == marker_set[2].tag or marker_set[0].tag == marker_set[3].tag or \
         marker_set[1].tag == marker_set[2].tag or marker_set[1].tag == marker_set[3].tag or \
         marker_set[2].tag == marker_set[3].tag:
         continue  # skip sets with duplicate colors (NOTE this assumes each face has 4 distinct colors)
      marker_seqs = list(permutations(marker_set, 4))
      #total_marker_seqs += len(marker_seqs)
      for marker_seq in marker_seqs:
        for name, face in cube_faces.iteritems():
          if np.any([marker.tag != cube_vertex_colors[vertex_idx] for marker, vertex_idx in zip(marker_seq, face)]):
            continue  # marker tags don't match face vertex colors
          face_points = np.float32([marker.imagePos for marker in marker_seq])
          if self.isCubeFaceValid(name, face_points):
            #valid_marker_seqs += 1
            face_marker_seqs.append((name, face, marker_seq, face_points))
    
    #self.logger.debug("total_marker_seqs = {}, valid_marker_seqs = {}".format(total_marker_seqs, valid_marker_seqs))
    if not face_marker_seqs:
      self.logger.debug("No matching (face, marker-sequence) pairs")
      return False
    
    # TODO compute scores for different faces based on whether there are edge pixels between vertices where expected (and what fraction of total length); also after reprojecting remaining points, whether actual image pixels match in color
    
    transforms = []  # TODO make this a named tuple
    for name, face, marker_seq, face_points in face_marker_seqs:
      face_vertices = np.float32([trackable.vertices[vertex_idx] for vertex_idx in face])
      retval, rvec, tvec = cv2.solvePnP(face_vertices, face_points, camera_params, dist_coeffs)
      if retval:
        transforms.append((name, rvec, tvec))
    
    if not transforms:
      self.logger.debug("No valid transforms could be computed")
      return False
    
    #self.logger.debug("{} transforms:\n{}".format(len(transforms), "\n".join(str(transform) for transform in transforms)))
    final_transform = transforms[0]
    if len(transforms) > 1:
      mean_tvec = np.mean([tvec for _, _, tvec in transforms], axis=0)
      #self.logger.debug("mean_tvec = {} ({} transforms)".format(mean_tvec, len(transforms)))
      consistent_transforms = [transform for transform in transforms if np.linalg.norm(transform[2] - mean_tvec, ord=2) < 10.0]
      #self.logger.debug("{} consistent transforms:\n{}".format(len(consistent_transforms), "\n".join(str(transform) for transform in consistent_transforms)))
      if len(consistent_transforms) == 0:
        if self.active:
          final_transform = sorted(transforms, key=lambda transform: np.linalg.norm(transform[2] - trackable.tvec, ord=2))[0]
      else:
        final_transform = ("/".join(transform[0] for transform in consistent_transforms), consistent_transforms[0][1], consistent_transforms[0][2])  # use rvec from the first consistent transform, and mean tvec (?)
    
    self.logger.debug("{} candidate transforms, final_transform:\n{}".format(len(transforms), "\n".join(str(x) for x in final_transform)))
    trackable.rvec = final_transform[1]
    trackable.tvec = final_transform[2]
    trackable.visible = True
    return True
  
  def isCubeFaceValid(self, face_name, face_points):
    # * Verify that these face points satisfy a known topographical structure (order)
    # NOTE Topographical structure is inherent in the order of face points (should be counter-clockwise)
    
    # ** Compute face centroid
    face_centroid = np.mean(face_points, axis=0)
    #print "face_centroid: {}".format(face_centroid)
    
    # ** [2D] Compute heading angles to each point assuming face normal is pointing outward through the screen
    heading_vecs = face_points - face_centroid
    headings = np.float32([ np.arctan2(vec[1], vec[0]) for vec in heading_vecs ])
    
    # ** [3D] Compute face plane normal and heading angles to each point around the normal (TODO)
    #face_normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[3] - face_vertices[0])
    #print "face_normal: {}".format(face_normal)
    
    # ** These headings should be in decreasing order (since they are counter-clockwise, and Y-axis is downwards)
    heading_diffs = headings - np.roll(headings, 1)
    heading_diffs = ((heading_diffs + pi) % (2 * pi)) - pi  # ensures angle wraparound is handled correctly
    #print "heading_diffs: {}".format(heading_diffs)
    if np.any(heading_diffs > 0):
      #self.logger.debug("Failed ordering check")
      return False  # if any heading angle difference is positive, skip this face
      
    # * Check if face points form a convex quadrilateral facing us
    # NOTE this eliminates the need for heading-based check above, doesn't it?
    #print "Face: {}".format(name)
    #print "face_points:\n{}".format(face_points)
    edge_vectors = face_points - np.roll(face_points, 1, axis=0)  # vectors between adjacent vertices (i.e. along edges)
    #print "edge_vectors:\n{}".format(edge_vectors)
    cross_prods = np.cross(edge_vectors, np.roll(edge_vectors, 1, axis=0))  # cross products of consecutive edge vectors
    #print "cross_prods: {}".format(cross_prods)
    if np.any(cross_prods < 0):
      #self.logger.debug("Failed convex quad, front-face check")
      return False  # if any cross product is negative, then quad is either non-convex or back-facing
    
    # * Check if opposite heading vector lengths and angles are almost equal (i.e. the quad is almost a rhombus)
    '''
    heading_lens = np.float32([np.hypot(vec[0], vec[1]) for vec in heading_vecs])
    #print "heading_lens: {}".format(heading_lens)
    if not (abs(heading_lens[0] - heading_lens[2]) / max(heading_lens[0], heading_lens[2]) < 0.5 \
        and abs(heading_lens[1] - heading_lens[3]) / max(heading_lens[1], heading_lens[3]) < 0.5 \
        and abs(heading_diffs[0] - heading_diffs[2]) / max(heading_diffs[0], heading_diffs[2]) < 0.1 \
        and abs(heading_diffs[1] - heading_diffs[3]) / max(heading_diffs[1], heading_diffs[3]) < 0.1):
        return False
    '''
    #print face_name
    #print "heading_vecs:\n{}".format(heading_vecs)
    heading_vecs = [vec / np.linalg.norm(vec, ord=2) for vec in heading_vecs]
    #print "norm. heading_vecs:\n{}".format(heading_vecs)
    #print "{} dot {} = {}".format(heading_vecs[0], heading_vecs[2], np.dot(heading_vecs[0], heading_vecs[2]))
    #print "{} dot {} = {}".format(heading_vecs[1], heading_vecs[3], np.dot(heading_vecs[1], heading_vecs[3]))
    if not (np.dot(heading_vecs[0], heading_vecs[2]) < -0.95 and np.dot(heading_vecs[1], heading_vecs[3]) < -0.95):
      #self.logger.debug("Failed rhombus check")
      return False
    
    if self.gui and doRenderFace:
      face_centroid_int = (int(face_centroid[0]), int(face_centroid[1]))
      cv2.circle(self.imageOut, face_centroid_int, 10, (0, 128, 0), -1)
      for point, heading in zip(face_points, headings):
        cv2.line(self.imageOut, face_centroid_int, (int(point[0]), int(point[1])), (0, 128, 0), 2)
        #label_pos = (face_centroid_int[0] + int(100 * np.cos(heading)), face_centroid_int[1] + int(100 * np.sin(heading)))
        #cv2.putText(self.imageOut, "{:.2f}".format(heading * 180 / pi), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
      cv2.putText(self.imageOut, face_name, face_centroid_int, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return True
  
  def trackVerticesRansac(self, trackable, activeMarkers):
    # NOTE len(activeMarkers) >= 4
    # [Tracking method 3] Try to match a set of (at least four) vertices that may or may not form a face
    '''
    # * Find different possible mappings between markers and blobs
    for blob in self.blobs: blob.used = False
    def findMappings(markers, i=0, pairs=[], mappings=[]):
      #self.logger.debug("i = {}, #markers = {}, #pairs = {}, #mappings = {}".format(i, len(markers), len(pairs), len(mappings)))
      if i == len(markers):
        mappings.append(pairs)
        #self.logger.debug("Mapping ({} pairs): ".format(len(pairs)) + ", ".join(("{} @ {}".format(pair[0].tag, pair[1].center) for pair in pairs)))
      else:
        for blob in markers[i].blobs:
          blobAppended = False
          if not blob.used:
            pair = (markers[i], blob)
            pairs.append(pair)
            blob.used = True
            blobAppended = True
          findMappings(markers, i + 1, list(pairs), mappings)
          if blobAppended:
            pairs.pop()
            blob.used = False
      return mappings
    
    mappings = []
    findMappings(activeMarkers, i=0, pairs=[], mappings=mappings)
    self.logger.debug("Mappings ({}):-".format(len(mappings)))
    for pairs in mappings:
      self.logger.debug("Mapping ({} pairs): ".format(len(pairs)) + ", ".join(("{} @ {}".format(pair[0].tag, pair[1].center_int) for pair in pairs)))
    
    # * For each mapping, compute a transform
    for pairs in mappings:
      if len(pairs) < 4: continue  # cv2.solvePnP() needs at least 4 point pairs
      #worldPositions = np.float32([marker.worldPos for marker in activeMarkers])
      #imagePositions = np.float32([marker.imagePos for marker in activeMarkers])
      worldPositions = np.float32([pair[0].worldPos for pair in pairs])
      imagePositions = np.float32([pair[1].center for pair in pairs])
      #self.logger.debug("[solvePnP] Input:-\nworldPositions:\n{}\nimagePositions:\n{}".format(worldPositions, imagePositions))
      retval, trackable.rvec, trackable.tvec = cv2.solvePnP(worldPositions, imagePositions, camera_params, dist_coeffs)
      #self.logger.debug("[solvePnP] Transform (retval: {}):-\nrvec:\n{}\ntvec:\n{}".format(retval, trackable.rvec, trackable.tvec))
      self.rvecRaw = trackable.rvec
      self.tvecRaw = trackable.tvec
      # ** Project a cube overlayed on top of video stream
      if self.gui and doRenderCube:
        cube_points, jacobian = cv2.projectPoints(self.cube_vertices, self.rvecRaw, self.tvecRaw, camera_params, dist_coeffs)
        cube_points = cube_points.reshape(-1, 2)  # remove nesting
        #self.logger.debug("Projected cube points:\n{}".format(cube_points))
        for u, v in self.cube_edges:
          if u < len(cube_points) and v < len(cube_points) and cube_points[u] is not None and cube_points[v] is not None:  # sanity check
            cv2.line(self.imageOut, (int(cube_points[u][0]), int(cube_points[u][1])), (int(cube_points[v][0]), int(cube_points[v][1])), (255, 255, 0), 2)
    '''
    
    # * Flatten marker-blob pairings to get two index-matched arrays
    worldPositions = np.float32([marker.worldPos for marker in activeMarkers])
    imagePositions = np.float32([marker.imagePos for marker in activeMarkers])
    '''
    worldPositions = []
    imagePositions = []
    for marker in activeMarkers:
      for blob in marker.blobs:
        worldPositions.append(marker.worldPos)
        imagePositions.append(blob.center)
    
    worldPositions = np.float32(worldPositions)
    imagePositions = np.float32(imagePositions)
    '''
    
    # * Find rotation and translation vectors from 3D-2D point correspondences
    self.logger.debug("Input:-\nworldPositions:\n{}\nimagePositions:\n{}".format(worldPositions, imagePositions))
    rvec, tvec, inliers = cv2.solvePnPRansac(worldPositions, imagePositions, camera_params, dist_coeffs, trackable.rvec, trackable.tvec, useExtrinsicGuess = trackable.visible)
    #self.logger.debug("Transform:-\nrvec:\n{}\ntvec:\n{}\ninliers:\n{}".format(rvec, tvec, inliers))
    
    # * If a valid transform is found, mark trackable as visible
    if rvec is not None and tvec is not None and inliers is not None:
      trackable.rvec = rvec
      trackable.tvec = tvec
      trackable.visible = True
      return True
    
    return False
  
  def trackCubeWithEdges(self, trackable, activeMarkers):
    # NOTE This method directly uses self.blobs and trackable.vertices instead of trackable.markers or activeMarkers
    
    # * Find edge pixels in the image by applying an appropriate HSV filter
    edgeMask = self.cubeEdgeFilter.apply(self.imageHSV)
    if self.gui: cv2.imshow("edge", edgeMask)
    edgeMask_img = cv.fromarray(edgeMask)  # cv.LineIterator() needs an IplImage/CvMat instead of a numpy array
    
    '''
    # * Find line segments in resulting binary edge mask
    lines = cv2.HoughLinesP(edgeMask, 5, np.radians(5.0), 100, minLineLength=100, maxLineGap=10)
    if lines is None:
      self.logger.info("No lines detected; bailing out")
      return False
    lines = lines[0]  # for some reason cv2.HoughLinesP() returns a numpy ndarray with shape: (1, <#lines>, 4)
    self.logger.info("{} line(s) detected".format(lines.shape[0]))
    if self.gui:
      for i in xrange(lines.shape[0]):
        line = lines[i]
        cv2.line(self.imageOut, (line[0], line[1]), (line[2], line[3]), (128, 0, 0), 1)
    '''
    
    '''
    # * Explore all possible assignments of blobs to markers and see which ones maintain connectivity
    blobs = list(self.blobs)
    if len(self.blobs) < len(trackable.vertices):
      blobs.extend([None] * (len(trackable.vertices) - len(self.blobs)))  # pad with None's
    
    blob_seqs = list(permutations(blobs, len(trackable.vertices)))
    blob_seq_scores = np.zeros(len(blob_seqs), dtype=np.float32)
    for blob_seq, blob_seq_score in zip(blob_seqs, blob_seq_scores):
      #self.logger.debug("Evaluating blob seq (len = {}): {}".format(len(blob_seq), blob_seq))
      self.logger.debug("Evaluating seq (len = {}): {}".format(len(blob_seq), ", ".join(blob.tag if blob is not None else "-" for blob in blob_seq)))
      
      # ** Check if this sequence is a possible match
      badSeq = False
      for j in xrange(len(blob_seq)):
        if blob_seq[j] is not None and blob_seq[j].tag != cube_vertex_colors[j]:
            badSeq = True
            self.logger.debug("Bad sequence: blob.tag = {}, cube_vertex_color = {}".format(blob_seq[j].tag, cube_vertex_colors[j]))
            break
      if badSeq: continue
      
      # ** Compute a connectivity score for this blob sequence
      for u, v in trackable.edges:
        self.logger.debug("Edge: ({}, {})".format(u, v))
        if blob_seq[u] is None or blob_seq[v] is None: continue  # we have no matching blobs for this vertex pair
        
        uPos = blob_seq[u].center
        vPos = blob_seq[v].center
        for line in lines:
          if np.inner(line[0:2] - uPos, line[0:2] - uPos) < 20 and np.inner(line[2:4] - vPos, line[2:4] - vPos) < 20 or \
             np.inner(line[0:2] - vPos, line[0:2] - vPos) < 20 and np.inner(line[2:4] - uPos, line[2:4] - uPos) < 20:
            blob_seq_score += 1
            if self.gui: cv2.line(self.imageOut, (line[0], line[1]), (line[2], line[3]), (255, 0, 255), 2)
            break  # one match found, that's good enough
    
    # ** Use computed blob sequence scores to pick best one
    best_seq_idx = np.argmax(blob_seq_scores)
    best_seq_score = blob_seq_scores[best_seq_idx]
    best_seq = blob_seqs[best_seq_idx]
    self.logger.info("Best blob sequence (score = {}): {}".format(best_seq_score, ", ".join(blob.tag if blob is not None else "-" for blob in best_seq)))
    '''
    
    '''
    # * Obtain blob connectivity graph using detected line segments
    blob_edges = []
    blob_adj = np.zeros(shape=(len(self.blobs), len(self.blobs)), dtype=np.uint8)  # adj. matrix representing blob connectivity
    for i in xrange(blob_adj.shape[0]):
      for j in xrange(i+1, blob_adj.shape[1]):
        for line in lines:
          dist_linePt1_i = np.inner(line[0:2] - self.blobs[i].center, line[0:2] - self.blobs[i].center)
          dist_linePt2_j = np.inner(line[2:4] - self.blobs[j].center, line[2:4] - self.blobs[j].center)
          dist_linePt1_j = np.inner(line[0:2] - self.blobs[j].center, line[0:2] - self.blobs[j].center)
          dist_linePt2_i = np.inner(line[2:4] - self.blobs[i].center, line[2:4] - self.blobs[i].center)
          #self.logger.debug("Line dists.: {}, {}, {}, {}".format(dist_linePt1_i, dist_linePt2_j, dist_linePt1_j, dist_linePt2_i))
          if (dist_linePt1_i < self.max_line_blob_distance_sq and dist_linePt2_j < self.max_line_blob_distance_sq) or \
             (dist_linePt1_j < self.max_line_blob_distance_sq and dist_linePt2_i < self.max_line_blob_distance_sq):
            blob_edges.append((i, j))
            blob_adj[i, j] = 1
            blob_adj[j, i] = 1
            #if self.gui: cv2.line(self.imageOut, (line[0], line[1]), (line[2], line[3]), (255, 0, 255), 2)  # show actual line segments
            if self.gui: cv2.line(self.imageOut, self.blobs[i].center_int, self.blobs[j].center_int, (255, 0, 255), 2)  # show blob links
            break  # one match found, that's good enough
    '''
    
    # * Obtain blob connectivity graph by tracing a line between pairs of blobs and studying the edge mask
    blob_edges = []
    blob_adj = np.zeros(shape=(len(self.blobs), len(self.blobs)), dtype=np.uint8)  # adj. matrix representing blob connectivity
    for i in xrange(blob_adj.shape[0]):
      for j in xrange(i+1, blob_adj.shape[1]):
        # ** For each blob pair, sample edgeMask pixels along the line connecting their centers, and find the sum (count)
        li = cv.InitLineIterator(edgeMask_img, self.blobs[i].center_int, self.blobs[j].center_int)
        count = sum(li) / 255
        dist = hypot(self.blobs[j].center[0] - self.blobs[i].center[0], self.blobs[j].center[1] - self.blobs[i].center[1])
        self.logger.debug("Line count ({} - {}) = {}, dist = {}".format(i, j, count, dist))
        
        # ** If count is at least some fraction of total distance between the centers, then there must be an edge there
        if count / dist >= 0.6:
          blob_edges.append((i, j))
          blob_adj[i, j] = 1
          blob_adj[j, i] = 1
          if self.gui: cv2.line(self.imageOut, self.blobs[i].center_int, self.blobs[j].center_int, (255, 0, 255), 2)
    
    # * Compute subgraph isomorphisms using blob adjacency matrix and cube adjacency matrix (as reference)
    blob_labels = [blob.tag for blob in self.blobs]
    self.logger.debug("Blob adjacency matrix:-\n{}".format(formatMatrix(blob_adj, blob_labels, blob_labels)))
    self.logger.debug("Cube adjacency matrix:-\n{}".format(formatMatrix(cube_adj, cube_vertex_colors, cube_vertex_colors)))
    
    algo = SubgraphIsomorphisms(blob_adj, blob_labels, cube_adj, cube_vertex_colors)
    isomorphisms = algo.run()
    if not isomorphisms:
      self.logger.info("No isomorphisms found; bailing out")
      return False
    self.logger.info("Found {} isomorphism(s)".format(len(isomorphisms)))
    if len(isomorphisms) > 1:
      self.logger.info("Isomorphisms:-\n" + "\n".join(formatMatrix(iso, blob_labels, cube_vertex_colors) for iso in isomorphisms))
    
    # * Select the best isomorphism (TODO find good criteria/use all isomorphisms and perform a fitness test later)
    mapping = isomorphisms[0]  # which one to use? why, the first one, of course!
    self.logger.info("Chosen isomorphism ({}x{}):\n{}".format(mapping.shape[0], mapping.shape[1], formatMatrix(mapping, blob_labels, cube_vertex_colors)))
    
    # * Using this blob-vertex mapping, extract index-matched object and image positions
    worldPositions = []
    imagePositions = []
    for i in xrange(len(self.blobs)):
      for j in xrange(len(trackable.vertices)):
        if mapping[i, j] == 1:
          worldPositions.append(trackable.vertices[j])
          imagePositions.append(self.blobs[i].center)
          if self.gui: cv2.putText(self.imageOut, str(j), self.blobs[i].center_int, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    worldPositions = np.float32(worldPositions)
    imagePositions = np.float32(imagePositions)
    
    if len(worldPositions) < 4:
      self.logger.info("Not enough 3D-2D point correspondences; bailing out")
      return False
    
    # * Find rotation and translation vectors from 3D-2D point correspondences
    self.logger.info("Input:-\nworldPositions:\n{}\nimagePositions:\n{}".format(worldPositions, imagePositions))
    # ** No RANSAC scheme
    retval, trackable.rvec, trackable.tvec = cv2.solvePnP(worldPositions, imagePositions, camera_params, dist_coeffs, useExtrinsicGuess = trackable.visible)
    self.logger.info("Transform (retval = {}):-\nrvec:\n{}\ntvec:\n{}".format(retval, trackable.rvec, trackable.tvec))
    trackable.visible = True
    return True
    
    '''
    # ** RANSAC scheme
    rvec, tvec, inliers = cv2.solvePnPRansac(worldPositions, imagePositions, camera_params, dist_coeffs, trackable.rvec, trackable.tvec, useExtrinsicGuess = trackable.visible)
    #self.logger.debug("Transform:-\nrvec:\n{}\ntvec:\n{}\ninliers:\n{}".format(rvec, tvec, inliers))
    
    # * If a valid transform is found, mark trackable as visible
    if rvec is not None and tvec is not None and inliers is not None:
      trackable.rvec = rvec
      trackable.tvec = tvec
      trackable.visible = True
      return True
    
    return False
    '''
  
  def updateTransform(self, rvec, tvec):
    #self.logger.debug("Transform:-\nrvec:\n{}\ntvec:\n{}".format(rvec, tvec))
    
    if self.active:
      tvec_diff_origin = np.linalg.norm(tvec - self.cube_origin, ord=2)
      tvec_diff_last = np.linalg.norm(tvec - self.tvecRaw, ord=2)
      rvec_diff_last = np.linalg.norm(((rvec - self.rvecRaw) + pi) % two_pi - pi, ord=2)  # norm of smallest angle difference along the 3 axes
      self.logger.debug("Dist. from origin: {}, from last pos.: {}; angle diff: {}".format(tvec_diff_origin, tvec_diff_last, rvec_diff_last))
      if tvec_diff_origin > self.tvec_maxdiff_origin or tvec_diff_last > self.tvec_maxdiff_last or rvec_diff_last > self.rvec_maxdiff_last:  # TODO check rvec_diff as well? (be careful with angle wraparound)
        self.logger.debug("Failed origin and continuity check")
        #self.smoothReset()
        return False
    
    self.rvecRaw = rvec.copy()
    self.tvecRaw = tvec.copy()
    if doSmoothPose:
      #self.rvecDiff = self.rvec - rvec
      self.rvecDiff = ((self.rvec - rvec) + pi) % two_pi - pi  # NOTE smaller difference between 2 angles; reverse order
      self.smoothUpdate()
    else:
      self.rvec = self.rvecRaw
      self.tvec = self.tvecRaw
    
    return True
  
  def smoothReset(self):
    """Initialize structs for storing transform samples over time for smoothing."""
    self.rvecRaws = np.zeros((3, self.num_smooth_samples), dtype=np.float32)
    self.rvecDiffs = np.zeros((3, self.num_smooth_samples), dtype=np.float32)
    self.tvecRaws = np.zeros((3, self.num_smooth_samples), dtype=np.float32)
    self.smoothIdx = 0
    self.rvec = np.zeros((3, 1), dtype=np.float32)
    self.tvec = np.zeros((3, 1), dtype=np.float32)
  
  def smoothUpdate(self):
    """Add current raw transform to samples, and compute smoothed (averaged) transform."""
    self.rvecRaws[:,self.smoothIdx:self.smoothIdx+1] = self.rvecRaw
    self.tvecRaws[:,self.smoothIdx:self.smoothIdx+1] = self.tvecRaw
    self.smoothIdx = (self.smoothIdx + 1) % self.num_smooth_samples  # circular buffer
    self.rvec = self.rvecRaw + np.mean(((self.rvecRaws - self.rvecRaw) + pi) % two_pi - pi, axis=1, keepdims=True)  # correct for angle wraparound: compute smallest angle difference with current raw rvec, take mean and add raw rvec back in
    self.tvec = np.mean(self.tvecRaws, axis=1, keepdims=True)
    # TODO Perform weighted averaging / take into account cube motion (Kalman filter)


class SubgraphIsomorphisms:
  """Given graphs as adjaceny matrices A and B (and corresponding vertex labels), find graph A in subgraphs of B."""
  # Based on: Ullmann, J. R., An Algorithm for Subgraph Isomorphism, JACM, Vol. 23, Iss. 1, pp. 31--42, 1976.
  #   Simple Enumeration algorithm (brute force)
  
  def __init__(self, A, A_labels, B, B_labels):
    # * Copy in matrices and labels
    self.A = A
    self.A_labels = A_labels
    self.B = B
    self.B_labels = B_labels
    self.logger = logging.getLogger(self.__class__.__name__)
  
  def run(self):
    # * Initialize isomorphism matrix by setting all possible mappings to 1
    self.pA = self.A.shape[0]  # number of "points" (vertices) in A
    self.pB = self.B.shape[0]  # number of "points" (vertices) in B
    self.M0 = np.zeros(shape=(self.pA, self.pB), dtype=np.uint8)
    for i in xrange(self.pA):
      for j in xrange(self.pB):
        if self.A_labels[i] == self.B_labels[j] and np.sum(self.B[:,j]) >= np.sum(self.A[:,i]):  # label and degree check
          self.M0[i, j] = 1
    self.logger.debug("Initial isomporphism matrix (M0):-\n{}".format(formatMatrix(self.M0, self.A_labels, self.B_labels)))
    
    self.F = np.zeros(self.pB, dtype=np.uint8)  # NOTE can be a true bit-vector
    self.H = np.zeros(self.pA, dtype=np.int8)
    self.M_ = [None] * self.pA  # M_[d] is matrix at depth d
    
    # * Run the algorithm!
    self.logger.debug("Starting algorithm")
    self.isomorphisms = []
    self.done = False
    self.step1()
    self.logger.debug("Algorithm complete (done = {}, #isomorphisms = {})".format(self.done, len(self.isomorphisms)))
    return self.isomorphisms
    
  def step1(self):
    self.logger.debug("Step 1")
    self.M = self.M0  # copy?
    self.d = 0  # we are 0-based; Ullmann's algorithm is 1-based
    self.H[self.d] = -1
    self.k = self.H[self.d]  # initialized here, so that we have self.k available
    self.F.fill(0)
    self.step2()
  
  def step2(self):
    self.logger.debug("Step 2: d = {}".format(self.d))
    if not np.any([self.M[self.d, j] == 1 and self.F[j] == 0 for j in xrange(self.pB)]):
      self.step7()
    else:
      self.M_[self.d] = self.M  # copy?
      self.k = self.H[self.d] if self.d == 0 else -1  # 0-based vs. 1-based
      self.step3()
      
  def step3(self):
    self.logger.debug("Step 3: d = {}, k = {}, M[d] = {}, F = {}".format(self.d, self.k, self.M[self.d], self.F))
    self.k = self.k + 1
    while self.M[self.d, self.k] == 0 or self.F[self.k] == 1:
      self.k = self.k + 1
    for j in xrange(self.pB):
      if j != self.k:
        self.M[self.d, j] = 0
    self.step4()
  
  def step4(self):
    self.logger.debug("Step 4")
    if self.d < (self.pA - 1):
      self.step6()
    else:
      self.isomorphisms.append(self.M)
      self.logger.debug("Candidate isomorphism (M):-")  # TODO Check if we have a valid isomorphism; if yes, add to list
      self.logger.debug(formatMatrix(self.M, self.A_labels, self.B_labels))
      self.step5()
  
  def step5(self):
    self.logger.debug("Step 5")
    if not np.any([self.M[self.d, j] == 1 and self.F[j] == 0 for j in xrange(self.k + 1, self.pB)]):
      self.step7()
    else:
      self.M = self.M_[self.d]  # copy?
      self.step3()
  
  def step6(self):
    self.logger.debug("Step 6")
    self.H[self.d] = self.k
    self.F[self.k] = 1
    self.d = self.d + 1
    self.step2()
    
  def step7(self):
    self.logger.debug("Step 7")
    if self.d == 0:
      self.done = True  # terminate algorithm
    else:
      self.F[self.k] = 0
      self.d = self.d - 1
      self.M = self.M_[self.d]
      self.k = self.H[self.d]
      self.step5()


def formatMatrix(mat, rowLabels, colLabels):  # TODO move to util?
  # TODO take into account max label widths?
  out = "\t{}\n".format("\t".join(colLabels))
  for i in xrange(mat.shape[0]):
    out += "{}\t{}\n".format(rowLabels[i], "\t".join(str(val) for val in mat[i]))
  return out


if __name__ == "__main__":
  options = { 'gui': True, 'debug': ('--debug' in sys.argv) }
  run(CubeTracker(options=options), gui=options['gui'], debug=options['debug'])
