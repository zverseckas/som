import numpy as np

"""
  Class represents a neuron used by SOM
"""
class Neuron(object):
  """
    Initializes a neuron with given coordinates `x` and `y` with
    `weight_count` of random weights
  """
  def __init__(self, x, y, weights_count):
    self._x = x
    self._y = y
    self._w = self.rand_weights(weights_count)

  """
    Generates `weight_count` of random weights from uniform distribution
  """
  @staticmethod
  def rand_weights(weights_count):
    return np.random.uniform(size=weights_count)

  """
    Returns a neron's position tuple
  """
  def position(self):
    return (self._x, self._y)

  """
    Returns a neron's weights
  """
  def weights(self):
    return self._w

  """
    Computes the Euclidean distance between a neuron's
    weights and a given `target_vector`
  """
  def euclidean_distance(self, target_vector):
    return np.sqrt(sum((self._w - target_vector)**2))

  """
    Determines whether a neuron is withing the `radius` of
    a given `center` neuron based on the weights vector
  """
  def is_within_radius(self, center, radius):
    c_x, c_y = center.position()
    return (self._x - c_x)**2 + (self._y - c_y)**2 < radius**2

  """
    Mutates the weights list based on the `target_vector` with respect
    to target's influance the learning rate
  """
  def update_weights(self, target_vector, influence, learning_rate):
    self._w += influence * learning_rate * (target_vector - self._w)
