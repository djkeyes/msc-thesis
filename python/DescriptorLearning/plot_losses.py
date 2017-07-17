

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This assumes data has one loss per line. To convert the caffe output to this format, consider running:
# grep -e ";" train-descriptors-32.out | grep -oE "[^ ]+$" > losses-32.txt

if __name__ == '__main__':
  losses = np.loadtxt("/home/daniel/losses-32.txt")
  t = np.arange(0, len(losses))

  # plt.scatter(t, losses, s=0.05)
  # plt.title('Training Loss Over Time, 32D')
  # plt.show()

  series = pd.Series(losses, t)
  window_size=100
  moving_med = series.rolling(window=window_size, center=False).median()
  plt.plot(moving_med.index[window_size-1:], moving_med[window_size-1:])
  plt.title('Training Loss Over Time, Rolling Median, 32D')
  plt.show()
