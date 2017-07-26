
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

actual = np.loadtxt('../cpp/build/actual.txt')
expected = np.loadtxt('../cpp/build/expected.txt')

plt.plot(expected[:,0], expected[:,1], 'g-')
plt.plot(actual[:,0], actual[:,1], 'r--')
for i in range(len(actual)):
  xs = [actual[i,0], expected[i,0]]
  ys = [actual[i,1], expected[i,1]]
  plt.plot(xs, ys, 'b-', lw=0.1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(expected[:,0], expected[:,1], expected[:,2], 'g-')
ax.plot(actual[:,0], actual[:,1], actual[:,2], 'r--')
for i in range(len(actual)):
  xs = [actual[i,0], expected[i,0]]
  ys = [actual[i,1], expected[i,1]]
  zs = [actual[i,2], expected[i,2]]
  ax.plot(xs, ys, zs, 'b-', lw=0.1)
plt.show()


