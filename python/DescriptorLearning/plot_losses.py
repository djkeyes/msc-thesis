

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This assumes data has one loss per line. To convert the caffe output to this format, consider running:
# grep -e ";" train-descriptors-32.out | grep -oE "[^ ]+$" > losses-32.txt

if __name__ == '__main__':
  losses = np.loadtxt("/home/daniel/losses-32-adam.txt")
  t = np.arange(0, len(losses))

  plt.scatter(t, losses, s=0.05)
  plt.title('Training Loss Over Time, 32D, ADAM')
  plt.ylim((0, 0.12))
  plt.show()

  series = pd.Series(losses, t)
  window_size=100
  moving_avg = series.rolling(window=window_size, center=False).mean()

  plt.plot(moving_avg.index[window_size - 1:], moving_avg[window_size - 1:])
  plt.title('Training Loss Over Time, Rolling Mean, 32D, ADAM')
  plt.ylim((0, 0.12))
  plt.show()

  # In our tests, gradients are accumulated over iter_size=12 evaulations of the loss function
  # Ergo group into groups of 12, then average. (Due to the extra printed lines during reshaping / other log output,
  # this might be off by one or something. hopefully doesn't matter.)
  # Also, the actual gradient is weighted by the number of point correspondences in each training sample. So for a more
  # meaningful analysis, we ought to compute the number of valid point correspondences per training pair and reweight
  #  by that.
  iter_size=12
  groups = series.groupby((series.index+iter_size-1)/iter_size)
  group_means = groups.mean()

  plt.plot(group_means)
  plt.title('Training Loss Over Time, 32D, ADAM')
  plt.ylim((0, 0.12))
  plt.show()
