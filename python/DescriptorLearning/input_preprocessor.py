
from collections import OrderedDict
from os import listdir
from os.path import isdir, join
import struct

from caffe.io import load_image

import numpy as np

import caffe

import matplotlib.pyplot as plt

def read_sparse_scene_coords(filename):
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

      result[row, col, 0] = x
      result[row, col, 1] = y
      result[row, col, 2] = z
  return result


def read_transform(filename):
  print 'reading from ', filename
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
    self.paths_by_directory = {}

  def setup(self, bottom, top):
    self.data_dir = self.param_str
    if len(bottom) != 0:
      raise Exception('must have zero inputs')
    if len(top) != 3:
      raise Exception('must have exactly three outputs')
    self.load_paths()
    self.data_generator = self.genTrainData()

  def reshape(self,bottom,top):
    top[2].reshape(1, 3, 4, 1)
    if(self.data_dir is not None):
      if not self.paths_by_directory:
        self.load_paths()
      # read one image to get the dimensions
      first_dir = self.paths_by_directory.keys()[0]
      first_dir_contents = self.paths_by_directory[first_dir]
      first_image = first_dir_contents[first_dir_contents.keys()[0]][0]
      img = load_image(join(self.paths_by_directory.keys()[0], first_image))
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
    for directory, frames in self.paths_by_directory.iteritems():
      # for now, just choose two adjacent frames
      # TODO: choose arbitrary covisible frames?
      prev_image = None
      prev_scene_coords = None
      prev_transform = None
      for _, content_tuple in frames.iteritems():
        image_file = content_tuple[0]
        scene_coord_file = content_tuple[1]
        transform_file = content_tuple[2]
        image = load_image(join(directory, image_file))
        # TODO: the other two
        scene_coords = read_sparse_scene_coords(join(directory, scene_coord_file)) # np.full([480, 640, 3], np.nan)

        print np.count_nonzero(~np.isnan(scene_coords))/2, ' non-nan vectors'
        transform = read_transform(join(directory, transform_file)) # np.zeros([3, 4])

        image = image.astype(np.float32)
        scene_coords = scene_coords.astype(np.float32)
        transform = transform.astype(np.float32)

        if np.any(np.isnan(transform)):
          # Sometimes transforms recorded from DSO can contain nan values. I
          # assume this is because the underlying optimization is
          # ill-conditioned
          # TODO(daniel): find the root cause
          continue

        if prev_image is not None:
          yield (prev_image, image, prev_scene_coords, scene_coords, prev_transform, transform)
        prev_image = image
        prev_scene_coords = scene_coords
        prev_transform = transform

  def forward(self,bottom,top):
    if self.data_dir is None:
      raise Exception('must specify a data directory')

    image_A, image_B, scene_coords_A, scene_coords_B, transform_A, transform_B = next(self.data_generator)
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
    self.paths_by_directory = {}
    directories = [join(self.data_dir, d) for d in listdir(self.data_dir) if isdir(join(self.data_dir, d))]
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
      self.paths_by_directory[dir] = frames_to_paths