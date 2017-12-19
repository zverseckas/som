from neuron import *

"""
  Self Organizing Map (SOM) algorithm implementation.

  Example:
    # Create a SOM instance of 20x20 dimensions with 4 input nodes
    som = SOM(shape=(20,20), input_size=4)
    # Load training data
    data = np.genfromtxt("train-data.csv")
    # Train the map with 1000 epochs
    som.train(data, epochs=1000)
"""
class SOM(object):
  """
    Initializes a SOM instance.
    `shape` is a tuple representing dimensions of a map,
    `input_size` determines a number of input nodes
  """
  def __init__(self, shape, input_size):
    self._shape = shape
    self._map = self.rand_map(shape, input_size)

  """
    Returns a neuron map of given `shape` with an
    `input_size` amount of input nodes
  """
  @staticmethod
  def rand_map(shape, input_size):
    return [
      Neuron(i, j, input_size)
      for i in range(shape[0])
      for j in range(shape[1])
    ]

  """
    Returns a normalized `vector`
  """
  @staticmethod
  def normalize(vector):
    return vector / sum(vector)

  """
    Returns a shape of a neuron map
  """
  def shape(self):
    return self._shape

  """
    Returns a neuron map
  """
  def map(self):
    return self._map

  """
    Finds the most similar neuron (`bmu`) to the `target_vector` in a map.
    Search is based on neuron weights and is computed using the
    Euclidean distance for vectors.
    Returns a tuple (`bmu`, `distance_from_bmu`, `distances_to_other_neurons`)
  """
  def find_bmu(self, target_vector):
    dists = [n.euclidean_distance(target_vector) for n in self._map]
    bmu = np.argmin(dists)
    return self._map[bmu], dists[bmu], dists

  """
    Returns the radius around the most similar neuron (`bmu`) depending on
    a current `timestamp` and a number of training `epochs`
  """
  def bmu_radius(self, timestamp, epochs):
    map_radius = max(self._shape) // 2
    time_const = epochs / np.log(map_radius)
    return map_radius * np.exp(-timestamp / time_const)

  """
    Returns the neurons that fall within the `radius` from a given node `bmu`
  """
  def neurons_near_bmu(self, bmu, radius):
    return [n for n in self._map if n.is_within_radius(bmu, radius)]

  """
    Performs a single training epoch for a given `target_vector` at a given
    `timestamp` with a provided number of `epochs` and a `learning_rate`.
    Note: Mutates the neuron weights.
  """
  def epoch(self, target_vector, timestamp, epochs, learning_rate):
    bmu, _, _ = self.find_bmu(target_vector)
    radius = self.bmu_radius(timestamp, epochs)
    learning_rate *= np.exp(-timestamp / epochs)
    for n in self.neurons_near_bmu(bmu, radius):
      influence = np.exp(-n.euclidean_distance(bmu.weights())**2 / (2 * radius**2))
      n.update_weights(target_vector, influence, learning_rate)

  """
    Trains a map by randomly selecting `data` vectors. Training is done
    in a given number of `epochs` with a given `learning_rate`
  """
  def train(self, data, epochs, learning_rate=0.1):
    for timestamp in range(epochs):
      rand_index = np.random.choice(len(data) - 1, 1)[0]
      self.epoch(self.normalize(data[rand_index]), timestamp, epochs, learning_rate)

  """
    Tests the map by guessing classes for testing `data`.
    Returns a tuple (`matrix_with_predicted_classes_and_local_quantization_errors`
    and an average `quantization_error` for the whole map
  """
  def test(self, data):
    result_map = np.empty(self._shape, dtype=tuple)
    quant_error = 0

    for target in data:
      bmu, min_dist, dists = self.find_bmu(self.normalize(target[:-1]))

      err = np.sqrt(sum([(min_dist - dist)**2 for dist in dists]))
      quant_error += err

      x, y = bmu.position()
      result_map[x][y] = (int(target[len(target) - 1]), err)

    return result_map, quant_error / len(data)
