import time

class Timer:
  def __init__(self, name):
    self.name = name
    self.cumtime = 0
    self.prev = 0
    self.lock = 0

  def start(self):
    if self.lock == 0:
      self.prev = time.time()
      self.lock = 1
    else:
      print("%s timer already recording" % self.name)

  def pause(self):
    if self.lock == 1:
      self.cumtime += time.time() - self.prev
      self.lock = 0
    else:
      print("%s timer not recording" % self.name)

  def reset(self):
    self.cumtime = 0
