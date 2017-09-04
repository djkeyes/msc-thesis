
from operator import itemgetter
from os import listdir
from os.path import join
import struct

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for projection='3d'

from input_preprocessor import read_transform, read_sparse_scene_coords


# train_dir = '/home/daniel/data/tmp/tum/sequence_01'
# train_dir = '/home/daniel/data/tmp/tum-dlt-withresults-caffe2/sequence_01'
# train_dir = '/home/daniel/data/tmp/tum-dlt-withresults/sequence_01'
# train_dir = '/home/daniel/data/tmp/tum-descriptors/sequence_01'
# train_dir = '/home/daniel/data/tmp/heads-descriptors/seq-01'
# train_dir = '/home/daniel/data/tmp/fire-descriptors/seq-01'
train_dir = '/home/daniel/data/tmp/redkitchen/seq-05'
# train_dir = '/home/daniel/data/tmp/heads/seq-02'
# train_dir = '/home/daniel/data/tmp/fire-morepts/seq-04'

def plot_translations_and_orientation_vecs(poses, ax, color, label=None):
  X, Y, Z, U, V, W = zip(*poses)
  # format for quiver:: x, y, z, dx, dy, dz
  ax.quiver(X, Y, Z, U, V, W, color=color)
  ax.plot(X, Y, Z, color + '-', label=label)

def verify_train_data():
  listdir(train_dir)

  scene_coords = sorted([file for file in listdir(train_dir) if file.startswith('sparse_scene_coords')])
  poses = sorted([file for file in listdir(train_dir) if file.startswith('pose')])

  if len(scene_coords) != len(poses):
    raise Exception('Mismatched number of images, scene coordinates, and poses')

  paths = []
  for i in range(len(poses)):
    frame_id = int(scene_coords[i][20:26])
    paths.append((frame_id, scene_coords[i], poses[i]))
  paths.sort(key=itemgetter(0))

  poses = []
  scene_coords = []
  num_processed = 0
  for frame_id, scene_coords_path, poses_path  in paths:
    pose = read_transform(join(train_dir, poses_path))
    R = pose[:, :3]
    t = pose[:, 3]

    R = R.T
    t = -R.dot(t)

    # for the scene coordinates, read by manually, to preserve sparseness
    filename = join(train_dir, scene_coords_path)
    # this is slow to render, so skip some
    if num_processed % 1 == 0:
      with open(filename, 'rb') as f:
        rows = struct.unpack('=I', f.read(4))[0]
        cols = struct.unpack('=I', f.read(4))[0]
        size = struct.unpack('=I', f.read(4))[0]
        for _ in range(size):
          row = struct.unpack('=H', f.read(2))[0]
          col = struct.unpack('=H', f.read(2))[0]
          x = struct.unpack('=f', f.read(4))[0]
          y = struct.unpack('=f', f.read(4))[0]
          z = struct.unpack('=f', f.read(4))[0]

          # set to true if files also have variance / observer cam pose
          if False:
            # read these, but just discard the result
            # inv depth (float), inv variance (float), pose (7 doubles)
            f.read(2 * 4 + 7 * 8)
          scene_coords.append((x, y, z))
    num_processed += 1


    # TODO: invert pose?
    poses.append(np.hstack((t, 0.025 * R.dot(t))))

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plot_translations_and_orientation_vecs(poses, ax, 'r')
  X, Y, Z = zip(*scene_coords)
  ax.plot(X, Y, Z, 'k.', ms=0.1)
  plt.axis('equal')
  # ax.set_xlim3d([-5, 5])
  # ax.set_ylim3d([-5, 5])
  # ax.set_zlim3d([-5, 5])
  ax.set_xlim3d([-2, 2])
  ax.set_ylim3d([-2, 2])
  ax.set_zlim3d([-2, 2])
  plt.title('Trajectory of train data ' + train_dir)
  plt.show()


if __name__ == "__main__":
  verify_train_data()