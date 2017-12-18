import sys
sys.path.append('../')

from som import *
import matplotlib.pyplot as plt

MAP_W = 30
MAP_H = 30
INPUT_SIZE = 4
EPOCHS = 900
LEARNING_RATE = 0.5

def prepare_data(file):
  return [
    np.genfromtxt(file, delimiter=",", usecols=(0, 1, 2, 3)),
    np.genfromtxt(file, delimiter=",")
  ]

def plot(result_map):
  plt.bone()
  markers = ["o", "s", ">"]
  colors  = ["r", "g", "b"]

  for i, row in enumerate(result_map):
    for j, col in enumerate(row):
      if col is None: continue
      w, err = col
      plt.plot(
        i + 0.5, j + 0.5, markers[w - 1],
        markerfacecolor='None',
        markeredgecolor=colors[w - 1],
        markersize=8,
        markeredgewidth=1,
      )

  plt.axis([0, MAP_W, 0, MAP_H])
  plt.show()

train_data, test_data = prepare_data("iris.csv")
som = SOM(shape=(MAP_W, MAP_H), input_size=INPUT_SIZE)
som.train(train_data, epochs=EPOCHS, learning_rate=LEARNING_RATE)
result_map, quantization_error = som.test(test_data)

plot(result_map)
print(quantization_error)
