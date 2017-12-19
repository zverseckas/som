import sys
sys.path.append('../')

from som import *
import matplotlib.pyplot as plt

# Map width
MAP_W = 30
# Map height
MAP_H = 30
# Number of input nodes
INPUT_SIZE = 4
# Number of learning epochs
EPOCHS = 900
# Initial learning rate
LEARNING_RATE = 0.5

"""
  Returns a tuple (`train_data`, `test_data`) from `file`
"""
def prepare_data(file):
  return [
    np.genfromtxt(file, delimiter=",", usecols=(0, 1, 2, 3)),
    np.genfromtxt(file, delimiter=",")
  ]

"""
  Draws a plot from a SOM results map
"""
def plot(result_map):
  plt.bone()
  markers = ["o", "s", "v"]
  colors  = ["r", "g", "b"]

  for i, row in enumerate(result_map):
    for j, col in enumerate(row):
      if col is None: continue
      w, err = col
      plt.plot(
        i + 0.5, j + 0.5, markers[w - 1],
        markerfacecolor='None',
        markeredgecolor=colors[w - 1],
        markersize=6,
        markeredgewidth=1,
      )

  plt.axis([0, MAP_W, 0, MAP_H])
  plt.show()

# Creates train and test data
train_data, test_data = prepare_data("iris.csv")
# Creates a SOM instance
som = SOM(shape=(MAP_W, MAP_H), input_size=INPUT_SIZE)

# Trains SOM
som.train(train_data, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# Tests SOM and plots results
result_map, quantization_error = som.test(test_data)
plot(result_map)
print(quantization_error)
