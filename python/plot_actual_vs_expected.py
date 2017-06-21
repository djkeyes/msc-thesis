
import numpy as np
import matplotlib.pyplot as plt

actual = np.loadtxt('../cpp/build/actual.txt')
expected = np.loadtxt('../cpp/build/expected.txt')

plt.plot(expected[:,0], expected[:,1], 'g-')
plt.plot(actual[:,0], actual[:,1], 'r--')
plt.show()


