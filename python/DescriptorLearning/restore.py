
import caffe

import matplotlib.pyplot as plt

if __name__ == '__main__':

  caffe.set_mode_cpu()
  # initialize caffe for gpu mode
  # caffe.set_mode_gpu()
  # caffe.set_device(0)

  # solver.restore('/home/daniel/experiments/denseCorrespondence/snap_iter_4000.solverstate')
  net = caffe.Net('./train_test.prototxt', './snap_iter_4000.caffemodel', caffe.TEST)

  net.forward()

  image = net.blobs['data'].data[0].transpose([1, 2, 0])
  plt.imshow(image)
  plt.show()
  image = net.blobs['upsample_1'].data[0].transpose([1, 2, 0])
  plt.imshow(image)
  plt.show()

