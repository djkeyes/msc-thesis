
import numpy as np
import matplotlib.pyplot as plt

actual = np.loadtxt('../cpp/build/actual.txt')
expected = np.loadtxt('../cpp/build/expected.txt')

plt.plot(expected[:,0], expected[:,1], 'g-')
plt.plot(actual[:,0], actual[:,1], 'r--')
for i in range(len(actual)):
  xs = [actual[i,0], expected[i,0]]
  ys = [actual[i,1], expected[i,1]]
  plt.plot(xs, ys, 'b-', lw=0.1)
plt.show()


