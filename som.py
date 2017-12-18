from neuron import *

class SOM(object):
  def __init__(self, shape, input_size):
    self._shape = shape
    self._map = self.rand_map(shape, input_size)

  @staticmethod
  def rand_map(shape, input_size):
    return [
      [Neuron(i, j, input_size) for i in range(shape[0])]
      for j in range(shape[1])
    ]
 
  def shape(self):
    return self._shape

  def map(self):
    return self._map


som = SOM(shape=(5,5), input_size=4)
print(som.map())