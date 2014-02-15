import time
import json
from threading import Thread
import zmq

from tool.Tool import Tool

class HapticPointer(Tool):
  """Wrapper class that reads in sensor values from a haptic pointing device over ZMQ."""
  
  sub_address = "tcp://localhost:60007"
  topic = "pose"
  
  def __init__(self):
    Tool.__init__(self)
    
    # * Initialize ZMQ context and open subscriber socket
    self.logger.debug("ZMQ version: {}, PyZMQ version: {}".format(zmq.zmq_version(), zmq.pyzmq_version()))
    self.zmqContext = zmq.Context()
    self.socket = self.zmqContext.socket(zmq.SUB)
    self.socket.connect(self.sub_address)
    self.logger.debug("Connected to {}".format(self.sub_address))
    
    # * Subscribe to appropriate topic
    self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
    self.logger.debug("Subscribed to topic \"{}\"".format(self.topic))
    time.sleep(0.005)  # mandatory sleep for ZMQ backend
    
    # * Initialize other members
    self.pose = None
    self.loop = True  # TODO ensure this is properly shared across threads
    
    # * Start sensing loop
    self.senseThread = Thread(target=self.senseLoop)
    self.senseThread.start()
    time.sleep(0.005)  # sleep to allow child thread to run
  
  def senseLoop(self):
    self.logger.info("[HapticPointer.senseLoop] Starting...")
    while self.loop:
      try:
        topic, data = self.socket.recv_multipart()
        self.logger.info("Topic: {}; Data: {}".format(topic, data))  # [debug]
        self.pose = data  # [debug: placeholder till decoding is ironed out]
        #self.pose = json.loads(data) # TODO correct JSON format and debug decoding
        #self.logger.info(self.pose)
      except KeyboardInterrupt:
        self.logger.info("[HapticPointer.senseLoop] Interrupted!")
        break
    self.logger.info("[HapticPointer.senseLoop] Done.")
  
  def close(self):
    self.loop = False
    self.logger.info("Waiting for HapticPointer.senseLoop thread to finish...")
    self.senseThread.join(1)
    if self.senseThread.is_alive():
      self.logger.warn("HapticPointer.senseLoop thread has not finished; please terminate using Ctrl+C")
    else:
      self.logger.info("HapticPointer.senseLoop thread finished.")
