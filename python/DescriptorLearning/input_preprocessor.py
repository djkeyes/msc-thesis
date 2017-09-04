
from collections import OrderedDict
from functools import partial
from os import listdir
from os.path import isdir, join
import random
import struct

from caffe.io import load_image

import numpy as np
from skimage.draw import line

import caffe

import matplotlib.pyplot as plt

def read_sparse_scene_coords(filename, has_variance=False):
  with open(filename, 'rb') as f:
    rows = struct.unpack('=I', f.read(4))[0]
    cols = struct.unpack('=I', f.read(4))[0]
    size = struct.unpack('=I', f.read(4))[0]
    # if this is slow, we could take pre-allocated memory as an argument
    result = np.full([rows, cols, 3], np.nan)
    for _ in range(size):
      row = struct.unpack('=H', f.read(2))[0]
      col = struct.unpack('=H', f.read(2))[0]
      x = struct.unpack('=f', f.read(4))[0]
      y = struct.unpack('=f', f.read(4))[0]
      z = struct.unpack('=f', f.read(4))[0]

      if has_variance:
        # read these, but just discard the result
        # inv depth (float), inv variance (float), pose (7 doubles)
        f.read(2*4 + 7*8)

      result[row, col, 0] = x
      result[row, col, 1] = y
      result[row, col, 2] = z
  return result


def read_transform(filename, from_trans_quat=True):
  if from_trans_quat:
    rows = 3
    cols = 4
    result = np.zeros([rows, cols])
    with open(filename, 'rb') as f:
      x = struct.unpack('=d', f.read(8))[0]
      y = struct.unpack('=d', f.read(8))[0]
      z = struct.unpack('=d', f.read(8))[0]
      qx = struct.unpack('=d', f.read(8))[0]
      qy = struct.unpack('=d', f.read(8))[0]
      qz = struct.unpack('=d', f.read(8))[0]
      qw = struct.unpack('=d', f.read(8))[0]

    R = np.asarray([
      [qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qz*qx+qw*qy)],
      [2*(qx*qy+qw*qz), qw*qw-qx*qx+qy*qy-qz*qz, 2*(qy*qz-qw*qx)],
      [2*(qz*qx-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy+qz*qz]
    ])
    t = np.asarray([x, y, z])

    t = t.reshape([3, 1])

    R = R.T
    t = -R.dot(t)
    result = np.hstack([R, t])
  else:
    rows = 3
    cols = 4
    result = np.zeros([rows, cols])
    with open(filename, 'rb') as f:
      # Eigen matrices are stored column-major by default
      for c in range(cols):
        for r in range(rows):
          result[r, c] = struct.unpack('=d', f.read(8))[0]

    R = result[:,0:3]
    t = result[:,3]
    t = t.reshape([3, 1])

    R = R.T
    t = -R.dot(t)
    result = np.hstack([R, t])

  return result

class ImageVertexTransformLayer(caffe.Layer):

  def __init__(self, layer_param):
    caffe.Layer.__init__(self, layer_param)
    self.data_dir = None
    self.data_loaders = []
    self.needs_reshape = True

  def setup(self, bottom, top):
    self.data_dir = self.param_str
    if len(bottom) != 0:
      raise Exception('must have zero inputs')
    if len(top) != 3:
      raise Exception('must have exactly three outputs')
    self.load_paths()
    self.data_generator = self.genTrainData()

  def reshape(self,bottom,top):
    if self.needs_reshape:
      self.needs_reshape = False
      top[2].reshape(1, 3, 4, 1)
      if(self.data_dir is not None):
        if len(self.data_loaders) == 0:
          self.load_paths()
        # read one image to get the dimensions
        for retry in range(5):
          content = self.data_loaders[0]()
          if content is not None:
            break
        img = content[0]
        channels = img.shape[2]
        height = img.shape[0]
        width = img.shape[1]
        top[0].reshape(2, channels, height, width)
        top[1].reshape(2, channels, height, width)
      else:
        # dummy values
        top[0].reshape(2, 3, 480, 640)
        top[1].reshape(2, 3, 480, 640)


  def genTrainData(self):
    while True:
      for loader in self.data_loaders:
        yield loader()

  def forward(self,bottom,top):
    if self.data_dir is None:
      raise Exception('must specify a data directory')

    for retries in range(100):
      content = next(self.data_generator)
      if content is not None:
        break
    image_A, scene_coords_A, transform_A, image_B, scene_coords_B, transform_B = content
    stereo = np.hstack([image_A, image_B])
    num_oob = 0
    count = 0
    for row in range(image_A.shape[0]):
      for col in range(image_A.shape[1]):
        coords = scene_coords_A[row, col, :]
        if np.isnan(coords[0]):
          continue
        count += 1

        projected_B = transform_B.dot(np.append(coords, 1))
        projected_A = transform_A.dot(np.append(coords, 1))
        # uB = int(round(projected_B[0]/projected_B[2]*277.34 + 312.234))
        # vB = int(round(projected_B[1]/projected_B[2]*291.402 + 239.777))
        # uA = int(round(projected_A[0]/projected_A[2]*277.34 + 312.234))
        # vA = int(round(projected_A[1]/projected_A[2]*291.402 + 239.777))
        uB = int(round(projected_B[0]/projected_B[2]*691.2 + 383.5))
        vB = int(round(projected_B[1]/projected_B[2]*691.2 + 207.5))
        uA = int(round(projected_A[0]/projected_A[2]*691.2 + 383.5))
        vA = int(round(projected_A[1]/projected_A[2]*691.2 + 207.5))

        if uB < 0 or vB < 0 or uB >= image_A.shape[1] or vB >= image_A.shape[0] or projected_B[2] <= 0:
          num_oob += 1
          continue
        if uA < 0 or vA < 0 or uA >= image_A.shape[1] or vA >= image_A.shape[0] or projected_A[2] <= 0:
          num_oob += 1
          continue

        if np.isnan(scene_coords_B[vB, uB][0]):
          num_oob += 1
          continue
        if count % 20 == 0:
          rr, cc = line(row, col, vB, image_A.shape[1] + uB)
          stereo[rr, cc, 0] = 0
          stereo[rr, cc, 1] = 1.0
          stereo[rr, cc, 2] = 0
          rr, cc = line(row, col, vA, uA)
          stereo[rr, cc, 0] = 1.0
          stereo[rr, cc, 1] = 0
          stereo[rr, cc, 2] = 0
    print "num oob: ", num_oob, "/", count
    plt.imshow(stereo)
    plt.show()


    image_A = image_A.transpose([2, 0, 1])
    image_B = image_B.transpose([2, 0, 1])

    scene_coords_A = scene_coords_A.transpose([2, 0, 1])
    scene_coords_B = scene_coords_B.transpose([2, 0, 1])

    transform_B = np.expand_dims(np.expand_dims(transform_B, axis=0), axis=3)

    top[0].data[...] = [image_A, image_B]
    top[1].data[...] = [scene_coords_A, scene_coords_B]
    # transform_A is unused, because the dense correspondence layer just projects points from world (stored in frame A) into frame B
    top[2].data[...] = transform_B

  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass

  def load_paths(self):
    paths_by_directory = {}
    directories = [join(self.data_dir, d) for d in listdir(self.data_dir) if isdir(join(self.data_dir, d)) and not d.endswith('results')]
    for dir in directories:
      images = sorted([file for file in listdir(dir) if file.startswith('image')])
      scene_coords = sorted([file for file in listdir(dir) if file.startswith('sparse_scene_coords')])
      poses = sorted([file for file in listdir(dir) if file.startswith('pose')])
      
      if len(images) != len(scene_coords) or len(scene_coords) != len(poses):
        raise Exception('Mismatched number of images, scene coordinates, and poses')

      frames_to_paths = OrderedDict()
      for i in range(len(images)):
        frame_id = int(images[i][6:12])
        frames_to_paths[frame_id] = (images[i], scene_coords[i], poses[i])
      paths_by_directory[dir] = frames_to_paths

    # create a bunch of functors to load training data, so we can invoke them
    # in shuffled order
    self.data_loaders = []
    for directory, frames in paths_by_directory.iteritems():
      adj_list = {}
      with open(join(directory, 'adjlist.txt')) as adj_list_file:
        adj_list_frames = []
        first = True
        line_num = 0
        for line in adj_list_file:
          values = [int(num) for num in line.split()]
          if first:
            N = values[0]
            adj_list_frames = values[1:]
            first = False
          else:
            M = values[0]
            adj_list[adj_list_frames[line_num]] = values[1:]
            line_num += 1

      for frame_id, content_tuple in frames.iteritems():
        image_file = content_tuple[0]
        scene_coord_file = content_tuple[1]
        transform_file = content_tuple[2]
        if frame_id not in adj_list:
          continue

        for adj_frame_id in adj_list[frame_id]:
          if adj_frame_id not in frames:
            continue
          adj_content_tuple = frames[adj_frame_id]
          adj_image_file = adj_content_tuple[0]
          adj_scene_coord_file = adj_content_tuple[1]
          adj_transform_file = adj_content_tuple[2]

          loader = partial(load_twice, directory, image_file, scene_coord_file, transform_file, adj_image_file,
                           adj_scene_coord_file, adj_transform_file)
          self.data_loaders.append(loader)
    random.shuffle(self.data_loaders)


def load_twice(directory, image_file, scene_coord_file, transform_file, adj_image_file, adj_scene_coord_file,
               adj_transform_file):
  first = load_data(directory, image_file, scene_coord_file, transform_file)
  second = load_data(directory, adj_image_file, adj_scene_coord_file, adj_transform_file)
  if first is None or second is None:
    return None
  return first + second

def load_data(directory, image_file, scene_coord_file, transform_file):
  print 'reading from directory ', directory, ' image ', image_file
  image = load_image(join(directory, image_file))
  scene_coords = read_sparse_scene_coords(join(directory, scene_coord_file))  # np.full([480, 640, 3], np.nan)
  print np.count_nonzero(~np.isnan(scene_coords)) / 3, ' non-nan vectors'
  transform = read_transform(join(directory, transform_file))  # np.zeros([3, 4])

  image = image.astype(np.float32)
  scene_coords = scene_coords.astype(np.float32)
  transform = transform.astype(np.float32)

  # I guess we should just be able to crop it, since the origin is (and hence the camera principle point is relative to)
  # the corner of the image?
  #n = 3
  #image = image[0:n*3*32, 0:n*4*32, :]
  #scene_coords = scene_coords[0:n*3*32, 0:n*4*32, :]

  if np.any(np.isnan(transform)):
    # Sometimes transforms recorded from DSO can contain nan values. I
    # assume this is because the underlying optimization is
    # ill-conditioned
    # TODO(daniel): find the root cause
    return None
  return (image, scene_coords, transform)