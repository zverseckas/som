import numpy as np

class Neuron(object):
  def __init__(self, x, y, weights_count):
    self._x = x
    self._y = y
    self._w = self.rand_weights(weights_count)

  @staticmethod
  def rand_weights(weights_count):
    return np.random.uniform(size=weights_count)

  def position(self):
    return (self._x, self._y)

  def weights(self):
    return self._w

  def euclidean_distance(self, target_vector):
    return np.sqrt(sum((self._w - target_vector) ** 2))
