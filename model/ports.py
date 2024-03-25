import copy

class InputPort:
  def __init__(self):
    self.data = None
  
  def receive(self, data):
    self.data = data

class OutputPort:
  def __init__(self):
    self.connected_ports = []
    self.num_ports = 0
  
  def connect(self, port):
    self.connected_ports.append(port)
    self.num_ports += 1
  
  def send(self, data):
    for port in range(self.num_ports):
      if port > 0:
        self.connected_ports[port].receive(copy.deepcopy(data))
      else:
        self.connected_ports[port].receive(data)