
import numpy as np
from skimage.draw import line
from scipy.misc import imresize
from skimage.exposure import rescale_intensity

import caffe

import matplotlib.pyplot as plt

if __name__ == '__main__':

  caffe.set_mode_cpu()
  # initialize caffe for gpu mode
  # caffe.set_mode_gpu()
  # caffe.set_device(0)

  # solver.restore('/home/daniel/experiments/denseCorrespondence/snap_iter_4000.solverstate')
  #net = caffe.Net('./train_test.prototxt', '/home/daniel/experiments/denseCorrespondence/snap_fullres_iter_12600.caffemodel', caffe.TEST)
  #net = caffe.Net('./train_test.prototxt', './snap_iter_19700.caffemodel', caffe.TEST)
  net = caffe.Net('./train_test.prototxt', '/home/daniel/experiments/denseCorrespondence/snap_fullres_labscene_iter_600.caffemodel', caffe.TEST)

  border = 2
  for i in range(10):
    net.forward()

    image_A = net.blobs['data'].data[0].transpose([1, 2, 0])
    image_B = net.blobs['data'].data[1].transpose([1, 2, 0])
    descr_A_up1 = net.blobs['upsample_1'].data[0].transpose([1, 2, 0])
    descr_B_up1 = net.blobs['upsample_1'].data[1].transpose([1, 2, 0])
    descr_A_fuse8 = net.blobs['fuse_s8'].data[0].transpose([1, 2, 0])
    descr_B_fuse8 = net.blobs['fuse_s8'].data[1].transpose([1, 2, 0])

    print 'descr_A_up1 min: ', np.min(descr_A_up1), ', median: ', np.median(descr_A_up1), 'max: ', np.max(descr_A_up1)
    print 'descr_B_up1 min: ', np.min(descr_B_up1), ', median: ', np.median(descr_B_up1), 'max: ', np.max(descr_B_up1)
    print 'descr_A_fuse8 min: ', np.min(descr_A_fuse8), ', median: ', np.median(descr_A_fuse8), 'max: ', np.max(descr_A_fuse8)
    print 'descr_B_fuse8 min: ', np.min(descr_B_fuse8), ', median: ', np.median(descr_B_fuse8), 'max: ', np.max(descr_B_fuse8)

    # rescale corresponding images in the same way
    range_values = (min(np.min(descr_A_up1), np.min(descr_B_up1)), max(np.max(descr_A_up1), np.max(descr_B_up1)))
    descr_A_up1 =rescale_intensity(descr_A_up1, in_range=range_values)
    descr_B_up1 =rescale_intensity(descr_B_up1, in_range=range_values)
    range_values = (min(np.min(descr_A_fuse8), np.min(descr_B_fuse8)), max(np.max(descr_A_fuse8), np.max(descr_B_fuse8)))
    descr_A_fuse8 =rescale_intensity(descr_A_fuse8, in_range=range_values)
    descr_B_fuse8 =rescale_intensity(descr_B_fuse8, in_range=range_values)

    descr_A_up1 = imresize(descr_A_up1, image_A.shape, interp='nearest')
    descr_B_up1 = imresize(descr_B_up1, image_B.shape, interp='nearest')
    descr_A_fuse8 = imresize(descr_A_fuse8, image_A.shape, interp='nearest')
    descr_B_fuse8 = imresize(descr_B_fuse8, image_B.shape, interp='nearest')



    scene_coords_A = net.blobs['vertex_data'].data[0].transpose([1, 2, 0])
    scene_coords_B = net.blobs['vertex_data'].data[1].transpose([1, 2, 0])
    transform_B = np.squeeze(net.blobs['transform_data'].data[0])

    stereo = np.hstack([image_A, np.ones([image_A.shape[0], border, 3]), image_B])
    stereo_desc_up1 = np.hstack([descr_A_up1, np.ones([image_A.shape[0], border, 3]), descr_B_up1])
    stereo_desc_fuse8 = np.hstack([descr_A_fuse8, np.ones([image_A.shape[0], border, 3]), descr_B_fuse8])
    count = 0
    for row in range(image_A.shape[0]):
      for col in range(image_A.shape[1]):
        coords = scene_coords_A[row, col, :]
        if np.isnan(coords[0]):
          continue

        projected_B = transform_B.dot(np.append(coords, 1))
        uB = int(round(projected_B[0]/projected_B[2]*277.34 + 312.234))
        vB = int(round(projected_B[1]/projected_B[2]*291.402 + 239.777))
        uA = col
        vA = row

        if uB < 0 or vB < 0 or uB >= image_B.shape[1] or vB >= image_B.shape[0] or projected_B[2] <= 0:
          continue

        rr, cc = line(row, col, vB, image_A.shape[1] + border + uB)
        rLoB = int(vB)
        cLoB = int(uB)
        rHiB = rLoB+1
        cHiB = cLoB+1
        if np.all(np.isnan(scene_coords_B[rLoB:rHiB+1,cLoB:cHiB+1,0])):
          continue

        # caffe also does a distance check here, after interpolating together the non-nan vertices

        count += 1
        if count % 50 == 0:
          stereo[rr, cc, 0] = 0.0
          stereo[rr, cc, 1] = 1.0
          stereo[rr, cc, 2] = 0.0

    plt.imshow(stereo)
    plt.title("Original images and feature correspondences (showing every 50th feature)")
    plt.show()
    plt.imshow(stereo_desc_up1)
    plt.title("Final upsampled features")
    plt.show()
    plt.imshow(stereo_desc_fuse8)
    plt.title("fuse_8 output (1/8 resolution)")
    plt.show()

