import time
import json
from threading import Thread
import zmq
from math import atan2, acos, pi
import numpy as np

import hommat as hm

from tool.Tool import Tool

class HapticPointer(Tool):
  """Wrapper class that reads in sensor values from a haptic pointing device over ZMQ."""
  
  # Define network interfaces for input (sub) and output (pub)
  sub_address = "tcp://localhost:60007"
  sub_topic = "trackStylus"
  sub_socket_linger = 2000  # ms; time to linger around after socket has been closed waiting for a recv()
  pub_address = "tcp://*:60017"
  pub_topic ="hapticControl"
  
  # Define transformation from device space to world (camera) space
  axis_pre_transform = np.float32(
      [[1, 0, 0, 0],
       [0, -1, 0, 0],
       [0, 0, -1, 80],
       [0, 0, 0, 1]])
  
  # TODO: Add further (post) transform for position, rotation, scale; build calibration routine
  position_scale = np.float32([1.0, 1.0, 1.0])
  position_offset = np.float32([0.0, 0.0, 80.0])  # move origin
  
  def __init__(self):
    Tool.__init__(self)
    
    # * Initialize ZMQ context and open subscriber, publisher sockets
    self.logger.debug("ZMQ version: {}, PyZMQ version: {}".format(zmq.zmq_version(), zmq.pyzmq_version()))
    # ** Context
    self.zmqContext = zmq.Context()
    # ** Subscriber
    self.subSocket = self.zmqContext.socket(zmq.SUB)
    self.subSocket.connect(self.sub_address)
    time.sleep(0.005)  # mandatory sleep for ZMQ backend
    self.logger.debug("[sub] Connected to {}".format(self.sub_address))
    # ** Subscriber topics for input messages
    self.subSocket.setsockopt(zmq.SUBSCRIBE, self.sub_topic)
    self.subSocket.setsockopt(zmq.LINGER, self.sub_socket_linger)
    self.logger.debug("[sub]Subscribed to topic \"{}\"".format(self.sub_topic))
    time.sleep(0.005)  # mandatory sleep for ZMQ backend
    # ** Publisher
    self.pubSocket = self.zmqContext.socket(zmq.PUB)
    self.pubSocket.bind(self.pub_address)
    time.sleep(0.005)  # mandatory sleep for ZMQ backend
    self.logger.debug("[pub] Bound to {}".format(self.pub_address))
    
    # * Initialize other members
    self.valid = False
    self.buttons = [0, 0]  # primary, secondary
    self.transform = hm.translation(hm.identity(), self.position_offset)
    #self.position = self.position_offset
    self.loop = True  # TODO ensure this is properly shared across threads
    
    # * Start sensing loop
    self.senseThread = Thread(target=self.senseLoop)
    self.senseThread.daemon = True  # to prevent indefinite wait on recv()
    self.senseThread.start()
    time.sleep(0.005)  # sleep to allow child thread to run
  
  def senseLoop(self):
    self.logger.info("[HapticPointer.senseLoop] Starting...")
    while self.loop:
      try:
        # Receieve data, parse JSON
        topic, data_str = self.subSocket.recv_multipart()
        #self.logger.info("Topic: {}; Data: {}".format(topic, data_str))  # [debug: raw incoming data]
        data = json.loads(data_str) # ensure correct JSON format (e.g. 1.0 instead of 1. for float numbers)
        #self.logger.info("JSON object: {}".format(data))  # [debug: JSON-decoded data]
        
        # Parse button state
        self.buttons = data['buttons']
        #self.logger.info("Buttons: {}".format(self.buttons))  # [debug: buttons]
        
        # Parse transform to get position, orientation and scale (separately, or use transform directly)
        self.transform = np.dot(self.axis_pre_transform, np.float32(data['transform']))
        #self.logger.info("Transform:\n{}".format(self.transform))  # [debug: transform]
        
        #self.position = self.transform[:3, 3] * self.position_scale + self.position_offset  # position: first 3 rows, last column
        #self.logger.info("Position: {}".format(self.position))  # [debug: position]
        # NOTE: orientation and scale together make up transform[0:3, 0:3]
        
        self.valid = True
      except KeyboardInterrupt:
        self.logger.info("[HapticPointer.senseLoop] Interrupted!")
        break
      except ValueError:
        self.logger.error("[HapticPointer.senseLoop] Bad JSON!")
        self.valid = False
    self.logger.info("[HapticPointer.senseLoop] Done.")
  
  def setFeedback(self, action, gain=0.0, magnitude=0.0):
    self.pubSocket.send(self.pub_topic, flags=zmq.SNDMORE)
    self.pubSocket.send("{} {} {}".format(action, gain, magnitude))
    # TODO: Change this to send a single magnitude value when other end is updated to use friction
  
  def close(self):
    self.loop = False
    self.logger.info("Waiting for HapticPointer.senseLoop thread to finish...")
    self.senseThread.join(1)
    #self.socket.close()  # TODO: check why this is causing an error
    #self.zmqContext.term()  # may cause problems; will be terminated when freed anyways
    if self.senseThread.is_alive():
      self.logger.warn("HapticPointer.senseLoop thread may not have finished; please terminate using Ctrl+C")
    else:
      self.logger.info("HapticPointer.senseLoop thread finished.")
