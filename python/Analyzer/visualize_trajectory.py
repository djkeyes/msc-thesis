from bisect import bisect_right
from os import listdir
from os.path import isdir, join
from enum import Enum

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for projection='3d'
import numpy as np
from collections import OrderedDict


class DriftCorrection(Enum):
  none = 1
  align_relative_poses = 2
  align_relative_poses_with_scale = 3


# note to self: you down-scaled the translations in cambridge by 100.
# rescale = 100
rescale = 1

# The TUM dataset is designed to be analyzed in a special way. The start and end of the sequence have high-fidelity
# mocap tracking (and empty poses in the middle), so trajectories can be split in half, aligned to the first half, and
# then compared to the second half for evaluation
is_tum = False
drift_correction = DriftCorrection.none

plotGroundTruth = False
plotVisualOdometry = False
showRelocalizationPlots = False


def read_results(directory):
  files = listdir(directory)
  ground_truth_files = [file for file in files if file.startswith('gt')]
  poses_from_vo_files = [file for file in files if file.startswith('vo')]
  estimate_files = [file for file in files if file.startswith('reloc')]

  ground_truth = OrderedDict()
  for file in ground_truth_files:
    id = int(file[3:5])
    trajectory = OrderedDict()
    contents = np.loadtxt(join(directory, file))
    if (len(contents.shape) == 1):
      contents = contents.reshape((1, -1))
    for i in range(0, contents.shape[0]):
      frameid = int(contents[i][0])
      t = rescale * contents[i][1:4].reshape(3, 1)
      R = contents[i][4:].reshape(3, 3)
      trajectory[frameid] = (R, t)
    ground_truth[id] = trajectory

  poses_from_vo = OrderedDict()
  for file in poses_from_vo_files:
    id = int(file[3:5])
    trajectory = OrderedDict()
    contents = np.loadtxt(join(directory, file))
    if (len(contents.shape) == 1):
      contents = contents.reshape((1, -1))
    for i in range(0, contents.shape[0]):
      frameid = int(contents[i][0])
      t = rescale * contents[i][1:4].reshape(3, 1)
      R = contents[i][4:].reshape(3, 3)
      trajectory[frameid] = (R, t)
    poses_from_vo[id] = trajectory

  estimates = OrderedDict()
  inlier_counts = OrderedDict()
  refframes = OrderedDict()
  for file in estimate_files:
    testid = int(file[6:8])
    trainid = int(file[18:20])

    trajectory = OrderedDict()
    traj_inliers_per_id = OrderedDict()
    traj_refframes = OrderedDict()
    contents = np.loadtxt(join(directory, file))
    if (len(contents.shape) == 1):
      contents = contents.reshape((1, -1))
    for i in range(0, contents.shape[0]):
      frameid = int(contents[i][0])
      ref_frameid = int(contents[i][1])
      num_inliers = int(contents[i][2])
      t = rescale * contents[i][3:6].reshape(3, 1)
      R = contents[i][6:].reshape(3, 3)
      trajectory[frameid] = (R, t)
      traj_inliers_per_id[frameid] = num_inliers
      traj_refframes[frameid] = ref_frameid
    estimates[(testid, trainid)] = trajectory
    inlier_counts[(testid, trainid)] = traj_inliers_per_id
    refframes[(testid, trainid)] = traj_refframes

  return (ground_truth, poses_from_vo, estimates, inlier_counts, refframes)


def get_translations_and_orientation_vecs(trajectory):
  result = np.array(
    [np.hstack((pose[1].T, 0.05 * pose[0].dot(np.array([[0], [0], [1]])).T, np.asarray([[frameid]]))) for frameid, pose
     in
     trajectory.items()]).squeeze()
  if len(result.shape) == 1:
    result = result.reshape([1, result.shape[0]])
  return result


def plot_translations_and_orientation_vecs(poses, ax, color, label=None, connect_lines=True):
  X, Y, Z, U, V, W, _ = zip(*poses)
  # format for quiver:: x, y, z, dx, dy, dz
  ax.quiver(X, Y, Z, U, V, W, color=color, label=label)
  if connect_lines:
    ax.plot(X, Y, Z, color + '-')


def compute_rot_dist_rad(R1, R2):
  R = R1.T.dot(R2)
  # convert rotation matrix to quaternion and return angle of axis-angle representation.
  if np.any(np.isnan(R)):
    return np.pi
  epsilon = 1e-3
  radicand = 1.0 + np.trace(R)
  if radicand > -epsilon and radicand < 4 + epsilon:
    # clip just in case it's barely out-of-range
    radicand = np.clip(radicand, 0, 4.0)
  else:
    # not sure what's up. probably corrupted gt measurements.
    return np.pi
  return np.arccos(0.5 * np.sqrt(radicand))


def compute_rot_dist_deg(R1, R2):
  rad = compute_rot_dist_rad(R1, R2)
  return np.rad2deg(rad)


def computeSim3Alignment(X, Y):
  """Compute a Sim(3) alignment between two sets of points.
     This computes a transform from x in X to y in Y so that s(Rx + t) = y holds. If the system is overconstrained 
     (more than 3 points), this minimizes the sum least-squared error; for this reason, this should not be used on 
     datasets which are known to contain outliers. Use a robust estimation method for that.     
     """
  if np.product(X.shape) == 0:
    return 1, np.identity(3), np.zeros((3,1))

  non_nan_idx = ~np.logical_or(np.any(np.isnan(X), axis=1), np.any(np.isnan(Y), axis=1))
  X = X[non_nan_idx, :]
  Y = Y[non_nan_idx, :]
  D = X.shape[0]
  meanX = np.mean(X, axis=0)
  meanY = np.mean(Y, axis=0)

  Xc = X - meanX
  Yc = Y - meanY

  A = Xc.T.dot(Yc)
  U, diag, V = np.linalg.svd(A)
  R = V.T.dot(U.T)
  if (np.linalg.det(R) < 0):
    V[2, :] *= -1
    R = V.T.dot(U.T)

  cov = 0
  var = 0
  for i in range(D):
    cov += R.dot(Xc[i, :]).dot(Yc[i, :])
    var += Xc[i, :].dot(Xc[i, :])
  s = cov / var

  t = meanY / s - R.dot(meanX)

  return s, R, t


def applySim3(orig_poses, s, R, t):
  if isinstance(orig_poses, dict):
    # assume we are given a dict id -> (R, t)
    result = OrderedDict()
    for (id, (R_cam, t_cam)) in orig_poses.items():
      result[id] = (R * R_cam, s * R * t_cam)
  else:
    # assume we have an array where each row is [ t u ], where u is a unit vector along the optical axis
    poses = s * (orig_poses[:, :3].dot(R.T) + t)
    orientations = orig_poses[:, 3:6].dot(R.T)
    frames = orig_poses[:, 6]
    result = np.hstack([poses, orientations, frames.reshape(frames.shape[0], 1)])
    if len(result.shape) == 1:
      result = result.reshape([1, result.shape[0]])
  return result


def alignRelativePosesToGT(reloc, reloc_refframes, scale_info, train_vo, train_ground_truth, normalize_scale):
  """Visual odometry drifts over time. Monocular visual odometry even has scale drift.
     To make a fair comparison, compute the relative pose between each relocated frame, at index i, and the VO frame
     used as reference, at index j. Apply this relative pose to the ground truth frame at index j. Optionally, normalize
     the magnitude of the relative translation so that it has the same magnitude as the true translation.
     In this way, we remove accumulated rotation and translation drift from visual odometry. Removing accumulated scale
     drift is tricker, and in the worst case, this overcompensates for the scale drift.
     If normalize_scale is False, scale_info should containing a single scaling constant. If normalize_scale is True,
      scale_info should countain the ground truth trajectory dict for the test sequence.
     """
  aligned_poses = OrderedDict()
  for frameid, ref_frameid in reloc_refframes.items():
    R_reloc, t_reloc = reloc[frameid]
    R_ref_vo, t_ref_vo = train_vo[ref_frameid]
    R_ref_gt, t_ref_gt = train_ground_truth[ref_frameid]

    R_rel = R_ref_vo.T.dot(R_reloc)
    t_rel = R_ref_vo.T.dot(t_reloc - t_ref_vo)

    if normalize_scale:
      reloc_ground_truth = scale_info
      t_rel_scale = np.linalg.norm(t_rel)
      # could have numerical precision issues if this is very small
      if t_rel_scale > 0.00001:
        t_rel_unit = t_rel / t_rel_scale
        # can evaluate this two ways. either we can make both translations equal
        # magnitude, or we can make the new translation be the closest point in
        # translation space (which would be the projection of GT onto rel)
        if False:
          scale = np.linalg.norm(reloc_ground_truth[frameid][1] - t_ref_gt)
        else:
          scale = (reloc_ground_truth[frameid][1] - t_ref_gt).T.dot(t_rel_unit)[0, 0]
        t_rel = t_rel_unit * scale
    else:
      t_rel *= 1  # scale_info

    R_aligned = R_ref_gt.dot(R_rel)
    t_aligned = R_ref_gt.dot(t_rel) + t_ref_gt
    aligned_poses[frameid] = (R_aligned, t_aligned)

  return aligned_poses


def main():
  # SIFT
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/chess/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/fire-orig/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/heads/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/office/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/pumpkin/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/redkitchen/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/stairs/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/greatcourt-orig/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/kingscollege/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/street/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/oldhospital/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/shopfacade/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/stmaryschurch/results')

  # DESCRIPTOR LEARNING
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/chess-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe/results-rat1.0')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/heads-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/office-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/pumpkin-caffe/results')
  ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/pumpkin-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/redkitchen-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/stairs-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/greatcourt-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/kingscollege-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/street-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/oldhospital-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/shopfacade-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/stmaryschurch-caffe/results')

  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe-all/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-sparse-caffe-48-obs/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe-all48/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe-all48-obs/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe-all/results-ransac200-0.99')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe-all/results-ransac200-0.99')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe/results-rat1.0')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe/results-rat0.99')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe/results-rat0.95')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-sparse-caffe/results-rat1.0')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-sparse-caffe/results-rat0.99')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/fire-caffe/results-rat1.0')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/fire-caffe-obs/results-rat1.0')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe-obs/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/heads-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/office-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/redkitchen-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/greatcourt-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/street-caffe/results')



  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/results-orig/fire-caffe/results')

  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/stairs-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/pumpkin/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/heads/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/redkitchen-caffe/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/tum/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/greatcourt/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/tum-dlt-withresults/results')
  # ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/tum-dlt-withresults-caffe2/results')

  # contents = np.loadtxt('/home/daniel/git/dso/build/result.txt')
  # i=0
  # for key in poses_from_vo[0].keys():
  #   poses_from_vo[0][key] = (poses_from_vo[0][key][0], contents[i][1:4].reshape(3, 1))
  #   i += 1

  colors = 'bgrcmy'

  s_by_seq = []
  R_by_seq = []
  t_by_seq = []
  for i in sorted(ground_truth.keys()):
    gt_poses = get_translations_and_orientation_vecs(ground_truth[i])
    vo_poses = get_translations_and_orientation_vecs(poses_from_vo[i])

    if is_tum:
      num_to_align = int(len(gt_poses) / 2)
    else:
      # num_to_align = len(poses)
      # Maybe it's better to align to the beginning of the frame?
      num_to_align = min(100, int(len(gt_poses) / 2))
    s, R, t = computeSim3Alignment(vo_poses[:num_to_align, :3], gt_poses[:num_to_align, :3])
    s_by_seq.append(s)
    R_by_seq.append(R)
    t_by_seq.append(t)

  if plotGroundTruth:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(ground_truth)):
      poses = get_translations_and_orientation_vecs(ground_truth[i])
      color = colors[i % len(colors)]
      plot_translations_and_orientation_vecs(poses, ax, color, 'seq-' + str(i))
    plt.title('Ground truth trajectories')
    plt.legend()
    plt.axis('equal')
    plt.show()

  if plotVisualOdometry:
    for i in poses_from_vo:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      poses = get_translations_and_orientation_vecs(poses_from_vo[i])
      gt_poses = get_translations_and_orientation_vecs(ground_truth[i])

      firstN = len(poses)
      poses = poses[:firstN, :]
      gt_poses = gt_poses[:firstN, :]

      aligned_poses = applySim3(poses, s_by_seq[i], R_by_seq[i], t_by_seq[i])
      plot_translations_and_orientation_vecs(aligned_poses, ax, 'r', 'Estimated from DSO')
      plot_translations_and_orientation_vecs(gt_poses, ax, 'k', 'Ground Truth')
      # to indicate start point
      ax.plot([aligned_poses[0, 0]], [aligned_poses[0, 1]], [aligned_poses[0, 2]], 'g.')
      ax.plot([gt_poses[0, 0]], [gt_poses[0, 1]], [gt_poses[0, 2]], 'g.')
      # corresponding frames
      for j in range(len(aligned_poses)):
        xs = [aligned_poses[j, 0], gt_poses[j, 0]]
        ys = [aligned_poses[j, 1], gt_poses[j, 1]]
        zs = [aligned_poses[j, 2], gt_poses[j, 2]]
        ax.plot(xs, ys, zs, 'b-', lw=0.1)
      plt.title('DSO-estimated trajectory, seq-' + str(i))
      plt.legend()
      plt.axis('equal')
      plt.show()

      # print(np.linalg.norm(aligned_poses[:, :3] - gt_poses[:, :3], axis=1).shape)
      # max = np.max(np.linalg.norm(aligned_poses[:, :3] - gt_poses[:, :3], axis=1))
      # print('max aligned translation error: ', max)

  if showRelocalizationPlots:
    reloc_fig = plt.figure()
    reloc_ax = reloc_fig.add_subplot(111, projection='3d')

    # plot training VO before everything else
    for train_id in poses_from_vo:
      has_tests = False
      for id in range(len(ground_truth)):
        if (train_id, id) in estimated:
          has_tests = True
          break
      if not has_tests:
        train_vo_poses = get_translations_and_orientation_vecs(poses_from_vo[train_id])
        aligned_train_vo_poses = applySim3(train_vo_poses, s_by_seq[train_id], R_by_seq[train_id], t_by_seq[train_id])
        plot_translations_and_orientation_vecs(aligned_train_vo_poses, reloc_ax, 'r', 'seq-' + str(train_id) + ' VO')

    all_train_distances = None
    for test_id in range(len(ground_truth)):

      # Might not have any queries for the train videos
      has_tests = False
      for train_id in range(len(ground_truth)):
        if (test_id, train_id) in estimated:
          has_tests = True
          break
      if not has_tests:
        continue

      test_gt_poses = get_translations_and_orientation_vecs(ground_truth[test_id])

      all_distances = None
      all_aligned_test_reloc_poses = None
      for train_id in range(len(ground_truth)):
        if (test_id, train_id) not in estimated:
          continue

        if drift_correction is not DriftCorrection.none:
          if drift_correction is DriftCorrection.align_relative_poses_with_scale:
            normalize_scale = True
            scale_info = ground_truth[test_id]
          elif drift_correction is DriftCorrection.align_relative_poses:
            normalize_scale = False
            scale_info = s_by_seq[train_id]
          else:
            assert (False)
          aligned_estimates = alignRelativePosesToGT(estimated[(test_id, train_id)],
                                                     refframes[(test_id, train_id)], scale_info,
                                                     poses_from_vo[train_id], ground_truth[train_id],
                                                     normalize_scale)
          aligned_test_reloc_poses = get_translations_and_orientation_vecs(aligned_estimates)
        else:
          test_reloc_poses = get_translations_and_orientation_vecs(estimated[(test_id, train_id)])
          aligned_test_reloc_poses = applySim3(test_reloc_poses, s_by_seq[train_id], R_by_seq[train_id],
                                               t_by_seq[train_id])

        gt_poses_by_frame = {gt_pose[6]: gt_pose[0:6] for gt_pose in test_gt_poses}
        distances = [np.linalg.norm(gt_poses_by_frame[row[6]][:3] - row[:3]) for row in aligned_test_reloc_poses]
        # distances = np.linalg.norm(test_gt_poses[non_default, :3] - aligned_test_reloc_poses[:, :3], axis=1)
        if all_distances is None:
          all_distances = distances
        else:
          all_distances = np.append(all_distances, distances)

        if all_aligned_test_reloc_poses is None:
          all_aligned_test_reloc_poses = aligned_test_reloc_poses
        else:
          all_aligned_test_reloc_poses = np.concatenate((all_aligned_test_reloc_poses, aligned_test_reloc_poses),
                                                        axis=0)

      if all_train_distances is None:
        all_train_distances = all_distances
      else:
        all_train_distances = np.append(all_train_distances, all_distances)

      all_distances = np.sort(all_distances)
      # fig = plt.figure()
      # ax = fig.add_subplot(111)
      # ax.step(all_distances, np.asarray(range(len(all_distances)))/len(all_distances))
      # plt.title('Cumulative error histogram, seq-' + str(test_id))
      # plt.xlabel('error')
      # plt.ylabel('Fraction of frames with less error')
      # plt.show()

      non_nan_idx = ~np.any(np.isnan(test_gt_poses), axis=1)
      test_gt_poses = test_gt_poses[non_nan_idx, :]
      # aligned_test_reloc_poses = aligned_test_reloc_poses[non_nan_idx, :]

      # Remove poses that are total outliers
      # to_keep = np.linalg.norm(aligned_test_reloc_poses[:, :3] - test_gt_poses[:, :3], axis=1) < 1
      # aligned_test_reloc_poses = aligned_test_reloc_poses[to_keep, :]

      # Remove poses that are total outliers
      # to_keep = np.linalg.norm(aligned_test_reloc_poses[:, :3], axis=1) < 10
      # aligned_test_reloc_poses = aligned_test_reloc_poses[to_keep, :]

      # plot half
      # aligned_test_reloc_poses = aligned_test_reloc_poses[:int(len(aligned_test_reloc_poses)/2-80), :]
      # aligned_test_reloc_poses = aligned_test_reloc_poses[int(len(aligned_test_reloc_poses)/2-40):, :]

      plot_translations_and_orientation_vecs(test_gt_poses, reloc_ax, 'k', 'seq-' + str(test_id) + ' GT')
      plot_translations_and_orientation_vecs(all_aligned_test_reloc_poses, reloc_ax, 'y',
                                             'seq-' + str(test_id) + ' est', False)

      # corresponding frames
      gt_poses_by_frame = {row[6]: row[0:6] for row in test_gt_poses}
      for aligned_reloc in all_aligned_test_reloc_poses:
        gt = gt_poses_by_frame[aligned_reloc[6]]
        xs = [aligned_reloc[0], gt[0]]
        ys = [aligned_reloc[1], gt[1]]
        zs = [aligned_reloc[2], gt[2]]
        reloc_ax.plot(xs, ys, zs, 'b-', lw=0.1)

    plt.axis('equal')
    if is_tum:
      reloc_ax.set_xlim3d([-5, 5])
      reloc_ax.set_ylim3d([-5, 5])
      reloc_ax.set_zlim3d([-5, 5])
    else:
      reloc_ax.set_xlim3d([-2, 2])
      reloc_ax.set_ylim3d([-2, 2])
      reloc_ax.set_zlim3d([-2, 2])
    plt.title('Relocalization')
    # plt.legend()
    plt.show()

  # for i in range(len(ground_truth)):
  #   vo_poses = get_translations_and_orientation_vecs(poses_from_vo[i])
  #   gt_poses = get_translations_and_orientation_vecs(ground_truth[i])
  #
  #   vo_poses = applySim3(vo_poses, s_by_seq[i], R_by_seq[i], t_by_seq[i])
  #
  #   vo_diffs = np.diff(vo_poses, axis=0)
  #   vo_rel_pose_dist_over_time = np.divide(np.linalg.norm(vo_diffs[:, :3], axis=1), vo_diffs[:,6])
  #   vo_rel_pose_timestamps = (vo_poses[1:, 6] + vo_poses[:vo_poses.shape[0]-1, 6])/2
  #
  #   gt_diffs = np.diff(gt_poses, axis=0)
  #   gt_rel_pose_dist_over_time = np.divide(np.linalg.norm(gt_diffs[:, :3], axis=1), gt_diffs[:,6])
  #   gt_rel_pose_timestamps = (gt_poses[1:, 6] + gt_poses[:gt_poses.shape[0]-1, 6])/2
  #   gt_rel_rot_over_time = calc_rel_angles(gt_poses)
  #
  #   fig = plt.figure()
  #   ax = fig.add_subplot(111)
  #   ax.plot(vo_rel_pose_timestamps, vo_rel_pose_dist_over_time, 'r', label='Aligned VO')
  #   ax.plot(gt_rel_pose_timestamps, gt_rel_pose_dist_over_time, 'k', label='GT')
  #   ax.plot(vo_rel_pose_timestamps, np.divide(gt_rel_rot_over_time, gt_rel_pose_dist_over_time)/500, 'b', label='GT rotation-to-translation ratio')
  #   plt.legend()
  #   plt.title('Relative translation between frames, seq-' + str(i))
  #   plt.show()



  # recompute all the distances, retaining angles
  all_distances_trans = np.array([])
  all_distances_deg = np.array([])
  for test_id in range(len(ground_truth)):
    # Might not have any queries for the train videos
    has_tests = False
    for train_id in range(len(ground_truth)):
      if (test_id, train_id) in estimated:
        has_tests = True
        break
    if not has_tests:
      continue

    for train_id in range(len(ground_truth)):
      if (test_id, train_id) not in estimated:
        continue

      if drift_correction is not DriftCorrection.none:
        if drift_correction is DriftCorrection.align_relative_poses_with_scale:
          normalize_scale = True
          scale_info = ground_truth[test_id]
        elif drift_correction is DriftCorrection.align_relative_poses:
          normalize_scale = False
          scale_info = s_by_seq[train_id]
        else:
          assert (False)
        aligned_estimates = alignRelativePosesToGT(estimated[(test_id, train_id)],
                                                   refframes[(test_id, train_id)], scale_info,
                                                   poses_from_vo[train_id], ground_truth[train_id],
                                                   normalize_scale)
      else:
        # aligned_estimates = applySim3(estimated[(test_id, train_id)], s_by_seq[train_id], R_by_seq[train_id], t_by_seq[train_id])
        # Just use the estimates directly...
        aligned_estimates = estimated[(test_id, train_id)]
      for (id, (R_est, t_est)) in aligned_estimates.items():
        (R_gt, t_gt) = ground_truth[test_id][id]
        dist_trans = np.linalg.norm(t_est - t_gt)
        if np.isnan(dist_trans):
          dist_trans = float('inf')
        all_distances_trans = np.append(all_distances_trans, [dist_trans])
        all_distances_deg = np.append(all_distances_deg, [compute_rot_dist_deg(R_est, R_gt)])

  num_frames = len(all_distances_trans)

  threshold_trans = 0.05
  threshold_rot_deg = 5

  threshold_fraction = np.count_nonzero(
    np.logical_and(all_distances_trans < threshold_trans, all_distances_deg < threshold_rot_deg))

  print('fraction below', threshold_trans, '(trans),', threshold_rot_deg, '(rot deg):', threshold_fraction / num_frames)

  all_distances_trans = np.sort(all_distances_trans)
  all_distances_deg = np.sort(all_distances_deg)

  median_error_trans = all_distances_trans[int(num_frames / 2)]
  median_error_rot_deg = all_distances_deg[int(num_frames / 2)]

  print('median error trans: ', median_error_trans)
  print('median error rot: ', median_error_rot_deg)

  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # ax.step(all_distances_trans, np.asarray(range(num_frames)) / num_frames)
  # plt.title('Cumulative error histogram, translation')
  # plt.xlabel('error')
  # plt.ylabel('Fraction of frames with less error')
  # plt.show()
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # ax.step(all_distances_deg, np.asarray(range(num_frames)) / num_frames)
  # plt.title('Cumulative error histogram, rotation (degrees)')
  # plt.xlabel('error')
  # plt.ylabel('Fraction of frames with less error')
  # plt.show()

  ground_truth, poses_from_vo, estimated, inlier_counts, refframes = read_results('/home/daniel/data/tmp/fire-caffe/results-rat1.0')
  ground_truth, poses_from_vo, estimated, inlier_counts2, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/fire-caffe/results-rat1.0')
  ground_truth, poses_from_vo, estimated, inlier_counts3, refframes = read_results('/home/daniel/data/tmp/results-aligned-sparser/fire-orig/results')

  all_inlier_counts = []
  for test_inlier_counts in inlier_counts.values():
    for count in test_inlier_counts.values():
      all_inlier_counts.append(count)
  all_inlier_counts = sorted(all_inlier_counts, reverse=True)

  all_inlier_counts2 = []
  for test_inlier_counts in inlier_counts2.values():
    for count in test_inlier_counts.values():
      all_inlier_counts2.append(count)
    all_inlier_counts2 = sorted(all_inlier_counts2, reverse=True)

  all_inlier_counts3 = []
  for test_inlier_counts in inlier_counts3.values():
    for count in test_inlier_counts.values():
      all_inlier_counts3.append(count)
      all_inlier_counts3 = sorted(all_inlier_counts3, reverse=True)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.step(all_inlier_counts, np.asarray(range(len(all_inlier_counts)))/len(all_inlier_counts), label='Dense Learned')
  ax.step(all_inlier_counts2, np.asarray(range(len(all_inlier_counts2)))/len(all_inlier_counts2), label='Sparse Learned')
  ax.step(all_inlier_counts3, np.asarray(range(len(all_inlier_counts3)))/len(all_inlier_counts3), label='Sparse SIFT')
  plt.title('Cumulative Inlier Distrubution')
  plt.xlabel('# Inliers')
  plt.ylabel('Fraction of frames with more inliers')
  plt.xlim([0,500])
  plt.legend()
  plt.show()


def calc_rel_angles(poses):
  # this isn't quite right, since it ignores rotation around the optical axis
  first = poses[1:, 3:6]
  second = poses[:poses.shape[0] - 1, 3:6]
  # assume all rows have same norm
  norm = np.linalg.norm(poses[0, 3:6])
  dotproducts = np.sum(np.multiply(first, second), axis=1) / norm / norm
  return np.arccos(np.clip(dotproducts, -1.0, 1.0))


if __name__ == "__main__":
  main()
