import sys
import os
import os.path as osp

import caffe
from caffe.io import load_image

import matplotlib.pyplot as plt
import numpy as np

def vis_square(data):
  """Take an array of shape (n, height, width) or (n, height, width, 3)
     and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

  # normalize data for display
  data = (data - data.min()) / (data.max() - data.min())

  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
              (0, 1), (0, 1))  # add some space between filters
             + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
  data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

  # tile the filters into an image
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

  plt.imshow(data);
  plt.axis('off')


if __name__ == '__main__':

  #caffe.set_mode_cpu()
  # initialize caffe for gpu mode
  caffe.set_mode_gpu()
  caffe.set_device(0)

  workdir = './'
  if not os.path.isdir(workdir):
      os.makedirs(workdir)

  solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
  #solver.restore('/home/daniel/experiments/denseCorrespondence/snap_iter_200.solverstate')

  solver.solve()
  #for i in range(5):
  #  solver.step(1)
    #filters = solver.net.params['conv1'][0].data
    #vis_square(filters.transpose(0, 2, 3, 1))
    #plt.show()

    #image = solver.net.blobs['data'].data[0].transpose([1, 2, 0])
    #plt.imshow(image)
    #plt.show()
    #image = solver.net.blobs['upsample_1'].data[0].transpose([1, 2, 0])
    #plt.imshow(image)
    #plt.show()
  solver.net.save('./trained.caffemodel')

