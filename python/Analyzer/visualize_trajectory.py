from os import listdir
from os.path import isdir, join

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # needed for projection='3d'
import numpy as np
from collections import OrderedDict



def read_results(directory):
  files = listdir(directory)
  ground_truth_files = [file for file in files if file.startswith('gt')]
  poses_from_vo_files= [file for file in files if file.startswith('vo')]
  estimate_files = [file for file in files if file.startswith('reloc')]

  ground_truth = OrderedDict()
  for file in ground_truth_files:
    id = int(file[3:5])
    trajectory = {}
    contents = np.loadtxt(join(directory, file))
    for i in range(0, contents.shape[0]):
      frameid = int(contents[i][0])
      t = contents[i][1:4].reshape(3,1)
      R = contents[i][4:].reshape(3,3)
      trajectory[frameid] = (R, t)
    ground_truth[id] = trajectory

  poses_from_vo = OrderedDict()
  for file in poses_from_vo_files:
    id = int(file[3:5])
    trajectory = {}
    contents = np.loadtxt(join(directory, file))
    for i in range(0, contents.shape[0]):
      frameid = int(contents[i][0])
      t = contents[i][1:4].reshape(3,1)
      R = contents[i][4:].reshape(3,3)
      trajectory[frameid] = (R, t)
    poses_from_vo[id] = trajectory

  estimates = OrderedDict()
  for file in estimate_files:
    testid = int(file[6:8])
    trainid = int(file[18:20])

    trajectory = {}
    contents = np.loadtxt(join(directory, file))
    for i in range(0, contents.shape[0]):
      frameid = int(contents[i][0])
      ref_frameid = int(contents[i][1])
      t = contents[i][2:5].reshape(3, 1)
      R = contents[i][5:].reshape(3, 3)
      trajectory[frameid] = (R, t)
    estimates[(testid, trainid)] = trajectory

  return (ground_truth, poses_from_vo, estimates)

def get_translations_and_orientation_vecs(trajectory):
  return np.array([np.hstack((pose[1].T, 0.025 * pose[0].dot(np.array([[0], [0], [1]])).T)) for frameid, pose in
           trajectory.items()]).squeeze()

def plot_translations_and_orientation_vecs(poses, ax, color, label=None):
    X, Y, Z, U, V, W = zip(*poses)
    # format for quiver:: x, y, z, dx, dy, dz
    ax.quiver(X, Y, Z, U, V, W, color=color)
    ax.plot(X, Y, Z, color + '-', label=label)


def main():
  ground_truth, poses_from_vo, estimated = read_results('/home/daniel/data/tmp/heads/results')

  colors = 'bgrcmy'

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for i in range(len(ground_truth)):
    poses = get_translations_and_orientation_vecs(ground_truth[i])
    color = colors[i % len(colors)]
    plot_translations_and_orientation_vecs(poses, ax, color, 'seq-' + str(i))
  plt.title('Ground truth trajectories')
  plt.legend()
  plt.show()

  for i in range(len(poses_from_vo)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    poses = get_translations_and_orientation_vecs(poses_from_vo[i])
    plot_translations_and_orientation_vecs(poses, ax, 'g')
    plt.title('DSO-estimated trajectory, seq-' + str(i))
    plt.show()

  for train_id in range(len(ground_truth)):
    train_gt_poses = get_translations_and_orientation_vecs(ground_truth[train_id])
    for test_id in range(len(ground_truth)):
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      test_gt_poses = get_translations_and_orientation_vecs(ground_truth[test_id])
      test_est_poses = get_translations_and_orientation_vecs(estimated[(test_id, train_id)])

      plot_translations_and_orientation_vecs(train_gt_poses, ax, 'b', 'seq-' + str(train_id) + ' GT')
      plot_translations_and_orientation_vecs(test_gt_poses, ax, 'g', 'seq-' + str(test_id) + ' GT')

      plot_translations_and_orientation_vecs(test_est_poses, ax, 'r', 'seq-' + str(test_id) + ' est')

      ax.set_xlim3d([-2, 2])
      ax.set_ylim3d([-2, 2])
      ax.set_zlim3d([-2, 2])
      plt.title('Relocalizing frame ' + str(test_id) + ' onto train sequence ' + str(train_id))
      plt.legend()
      plt.show()


if __name__ == "__main__":
  main()



