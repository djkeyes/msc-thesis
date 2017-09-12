from collections import OrderedDict
from collections import deque
import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from input_preprocessor import read_transform, read_sparse_scene_coords


def scene_coords_to_depth(scene_coords, pose):
  # we already have (u,v) coordinates, all we need is the depth
  # fortunately, that's easy to compute. It's just the z-value in the local coordinate frame
  R = pose[:3, :3]
  t = pose[:, 3, np.newaxis]

  # cam2world -> world2cam  
  Rinv = R.T
  tinv = -R.dot(t)

  depth = -1 * np.ones((scene_coords.shape[0:2]))
  for r in range(depth.shape[0]):
    for c in range(depth.shape[1]):
      if not np.any(np.isnan(scene_coords[r, c, :])):
        local = Rinv.dot(scene_coords[r, c, :, np.newaxis]) + tinv
        depth[r, c] = local[2]
  return depth


def inpaint(semidense_depth, initial_filler=None):
  has_initial_depth = semidense_depth > 0

  # assume min and max ranges. in general this is rather fragile.
  semidense_depth = np.clip(semidense_depth, 0, 3)

  # to speed up convergence, fill empty pixels with an initial guess
  if initial_filler is not None:
    semidense_depth = initial_filler(semidense_depth, has_initial_depth)

  sigma = 1.5
  tau = 1.5
  lamd = 5
  theta = 1


  Ix = semidense_depth
  Ix_bar = Ix.copy()
  Iy = np.zeros((Ix_bar.shape[0], Ix_bar.shape[1], 2))
  for t in range(2000):
    if t % 50 == 0:
      plt.imshow(np.clip(Ix, 0, 3))
      plt.title('Inpainted depth at t=' + str(t))
      plt.show()

    # update y
    # forward differences w/ Neumann
    gradXBar_0 = np.vstack((np.diff(Ix_bar, axis=0), np.zeros((1, Ix_bar.shape[1]))))
    gradXBar_1 = np.hstack((np.diff(Ix_bar, axis=1), np.zeros((Ix_bar.shape[0], 1))))
    # # centered differences
    # (gradXBar_0, gradXBar_1) = np.gradient(Ix_bar)
    gradXBar = np.stack((gradXBar_0, gradXBar_1), axis=2)
    proxFArg = Iy + sigma * gradXBar
    norm = 1 / np.maximum(1, np.linalg.norm(proxFArg, axis=2))
    Iy = proxFArg * norm[:,:,np.newaxis]
    # plt.imshow(norm)
    # plt.show()
    # norm = cat(4, norm, norm);


    # update x
    # backward differences w/ Neumann
    dy_0 = np.vstack((np.zeros((1, Ix_bar.shape[1])), np.diff(Iy[:,:,0], axis=0)))
    dy_1 = np.hstack((np.zeros((Ix_bar.shape[0], 1)), np.diff(Iy[:,:,1], axis=1)))
    # # centered differences
    # dy_0 = np.gradient(Iy[:,:,0], axis=0)
    # dy_1 = np.gradient(Iy[:,:,1], axis=1)
    divY = dy_0 + dy_1
    proxGArg = Ix + tau * divY

    Ix_new = proxGArg * (~has_initial_depth) + ((proxGArg + tau * lamd * Ix) / (1 + tau * lamd )) * has_initial_depth

    # update x_bar
    Ix_bar = Ix_new + theta * (Ix_new - Ix)
    Ix = Ix_new

  semidense_depth = Ix
  return semidense_depth


def meanFill(semidense_depth, has_initial_depth):
  """
  Assign depths the mean value of known depths. This modifies semidense_depth in-place.
  :param semidense_depth: semi-dense depth map 
  :param has_initial_depth: boolean array indicating which values to keep constant
  :return: the modified semidense_depth
  """
  mean = np.mean(semidense_depth[has_initial_depth])
  semidense_depth[~has_initial_depth] = mean
  return semidense_depth


def medianFill(semidense_depth, has_initial_depth):
  """
  Assign depths the median value of known depths. This modifies semidense_depth in-place.
  :param semidense_depth: semi-dense depth map 
  :param has_initial_depth: boolean array indicating which values to keep constant
  :return: the modified semidense_depth
  """
  median = np.median(semidense_depth[has_initial_depth])
  semidense_depth[~has_initial_depth] = median
  return semidense_depth


def nnFill(semidense_depth, has_initial_depth):
  """
  Assign depths to their nearest neighbor in the image plane via BFS. This modifies semidense_depth in-place.
  :param semidense_depth: semi-dense depth map 
  :param has_initial_depth: boolean array indicating which values to keep constant
  :return: the modified semidense_depth
  """
  return decayToMeanFill(semidense_depth, has_initial_depth, 1.0)


def decayToMeanFill(semidense_depth, has_initial_depth, alpha):
  """
  Assign depths using BFS, but at each step the depth decays by alpha toward the mean depth. This modifies semidense_depth in-place.
  :param semidense_depth: semi-dense depth map 
  :param has_initial_depth: boolean array indicating which values to keep constant
  :param alpha: the decay rate, so that next_depth = alpha*cur_depth + (1-alpha)*mean
  :return: the modified semidense_depth
  """
  mean = np.mean(semidense_depth[has_initial_depth])
  nonzero_coords = zip(*np.nonzero(has_initial_depth))
  queue = deque([(row, col, semidense_depth[row, col]) for (row, col) in nonzero_coords])
  visited = has_initial_depth.copy()
  while queue:
    (cur_row, cur_col, cur_depth) = queue.popleft()
    next_depth = alpha * cur_depth + (1 - alpha) * mean
    neighbors = []
    if cur_row > 0:
      neighbors.append((cur_row - 1, cur_col))
    if cur_col > 0:
      neighbors.append((cur_row, cur_col - 1))
    if cur_row < semidense_depth.shape[0] - 1:
      neighbors.append((cur_row + 1, cur_col))
    if cur_col < semidense_depth.shape[1] - 1:
      neighbors.append((cur_row, cur_col + 1))
    for (next_row, next_col) in neighbors:
      if not visited[next_row, next_col]:
        visited[next_row, next_col] = True
        semidense_depth[next_row, next_col] = next_depth
        queue.append((next_row, next_col, next_depth))
  return semidense_depth


def main():
  # Assumes visual odometry has already been performed, and this sequence is a training sequence (non-garbage depth values)
  sequence_dir = '/home/daniel/data/tmp/fire-caffe/seq-02'

  # filler = None
  # filler = meanFill
  # filler = medianFill
  filler = nnFill
  # filler = partial(decayToMeanFill, alpha=0.9)

  scene_coords = sorted([file for file in os.listdir(sequence_dir) if file.startswith('sparse_scene_coords')])
  poses = sorted([file for file in os.listdir(sequence_dir) if file.startswith('pose')])

  if len(scene_coords) != len(poses):
    raise Exception('Mismatched number of scene coordinates and poses')

  frames_to_paths = OrderedDict()
  for i in range(len(scene_coords)):
    frame_id = int(scene_coords[i][len('sparse_scene_coords_'):len('sparse_scene_coords_XXXXXX')])
    frames_to_paths[frame_id] = (scene_coords[i], poses[i])

  for (frame_id, (scene_coord_file, pose_file)) in frames_to_paths.iteritems():
    scene_coords = read_sparse_scene_coords(os.path.join(sequence_dir, scene_coord_file))
    pose = read_transform(os.path.join(sequence_dir, pose_file))
    semidense_depth = scene_coords_to_depth(scene_coords, pose)

    # ostensibly this is in meters, so we shouldn't get very large values
    plt.imshow(np.clip(semidense_depth, 0, 3), cmap='jet')
    plt.title('Semi-Dense Depth')
    plt.show()

    dense_depth = inpaint(semidense_depth, filler)
    plt.imshow(np.clip(dense_depth, 0, 3), cmap='jet')
    plt.title('In-Painted Dense Depth')
    plt.show()


if __name__ == "__main__":
  main()
