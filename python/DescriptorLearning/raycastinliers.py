import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
import random
import struct
from math import ceil, sqrt

W = 640
H = 480
# W = 100
# H = 100

# Radius to assign a vertex to an image coordinate, in square pixels
# Also used when we're only checking semi-dense coords, if the current vertex is empty
vertex_projection_tolerance = 2 * (2 ** 2)
# radius to consider two vertices the same, in mm
visibility_tolerance = 10
# descriptors within this radius to a true positive are also considered a true positive
equivilant_coord_dist_sq = 2 * (5 ** 2)

# assumes a hardcoded camera model
fx = 533.377
fy = 535.726
cx = 311.928
cy = 243.044
K = np.asarray([
  [fx, 0, cx],
  [0, fy, cy],
  [0, 0, 1]
])
Kinv = np.linalg.inv(K)


def getSelfProjectingVertices(mesh, cam2world, coords_to_check=None):
  # For each pixel in the camera, raycast a line from the camera center through that pixel into the world
  # If the ray intersects a polygon, check each vertex of the polygon. If that vertex re-projects back
  # onto the pixel, return the vertex id

  R = cam2world[0]
  t = cam2world[1]

  Rinv = R.T
  tinv = -Rinv.dot(t)

  if coords_to_check is None:
    size = W * H
  else:
    size = len(coords_to_check)
  origins = np.matlib.repmat(t.T, size, 1)
  directions = np.empty((size, 3))
  ray_index_to_coords = {}
  rays_so_far = 0
  if coords_to_check is None:
    for r in range(H):
      for c in range(W):
        p = np.asarray([c, r, 1]).reshape(3, 1)
        directions[rays_so_far, :] = (R.dot(Kinv.dot(p))).T
        ray_index_to_coords[rays_so_far] = (r, c)
        rays_so_far += 1
  else:
    for (row, col) in coords_to_check:
      p = np.asarray([col, row, 1]).reshape(3, 1)
      directions[rays_so_far, :] = (R.dot(Kinv.dot(p))).T
      ray_index_to_coords[rays_so_far] = (row, col)
      rays_so_far += 1

  print 'Finding intersections...'
  index_tri, index_ray = mesh.ray.intersects_id(origins, directions, multiple_hits=False)
  print 'Complete!'

  print 'Found', len(index_ray), ' hits!'

  visible_verts = np.zeros((H, W, 3))
  for ray_idx, tri_idx in zip(index_ray, index_tri):
    row = ray_index_to_coords[ray_idx][0]
    col = ray_index_to_coords[ray_idx][1]
    verts = mesh.triangles[tri_idx]
    min_distsq = vertex_projection_tolerance
    for i in range(3):
      vert = verts[i, :].reshape(3, 1)
      # this operation is really slow for some reason?
      # vert = mesh.vertices[mesh.faces[tri_idx][i]].reshape(3,1)
      # camera projection
      local = Rinv.dot(vert) + tinv
      proj = K.dot(local)
      z = proj[2]
      proj = proj[0:2] / z
      row_proj = proj[1]
      col_proj = proj[0]
      dr = row_proj - row
      dc = col_proj - col
      distsq = dr * dr + dc * dc
      if distsq <= min_distsq:
        visible_verts[row, col, :] = np.squeeze(vert)
        min_distsq = distsq
  return visible_verts


def check_visible(mesh, cam2world, vertex, coords_to_check=None):
  R = cam2world[0]
  t = cam2world[1]

  Rinv = R.T
  tinv = -Rinv.dot(t)

  vertex = vertex.reshape((3, 1))
  # first check it's actually in the camera frustum
  local = Rinv.dot(vertex) + tinv
  proj = K.dot(local)
  z = proj[2]
  if z <= 0:
    return False
  proj = proj[0:2] / z
  row_proj = proj[1]
  col_proj = proj[0]
  if row_proj < 0 or col_proj < 0 or row_proj >= H or col_proj >= W:
    return False

  origin = t
  direction = vertex - t
  direction /= np.linalg.norm(direction)

  index_tri, index_ray = mesh.ray.intersects_id(origin.reshape((1, 3)), direction.reshape((1, 3)), multiple_hits=False)

  if len(index_tri) == 0:
    return False

  if not coords_to_check:
    verts = mesh.triangles[index_tri[0]]
    for i in range(3):
      candidate = verts[i, :].reshape(3, 1)
      if np.linalg.norm(candidate - vertex) < visibility_tolerance:
        return (int(round(np.asscalar(row_proj))), int(round(np.asscalar(col_proj))))
  else:
    # skip it if it doesn't land on a vertex
    coord = (int(round(np.asscalar(row_proj))), int(round(np.asscalar(col_proj))))
    if coord not in coords_to_check:
      return False

    verts = mesh.triangles[index_tri[0]]
    for i in range(3):
      candidate = verts[i, :].reshape(3, 1)
      if np.linalg.norm(candidate - vertex) < visibility_tolerance:
        return coord
        # TODO: do several raycasts from all the coords within the equivilance distance
        # # Anywhere within a window of the true projection is acceptable
        # # We would like the returned point to be close to the projected point
        # # but only if it's also close to one of the vertices
        # window_rad = int(ceil(sqrt(equivilant_coord_dist_sq))) + 1
        # true_row = np.asscalar(row_proj)
        # true_col = np.asscalar(col_proj)
        # row = round(true_row)
        # col = round(true_col)
        # # there are more efficient ways to do this search, but this is okay
        # for dr in range(-window_rad, window_rad+1):
        #   for dc in range(-window_rad, window_rad+1):
        #     cur_row = row + dr
        #     cur_col = col + dc
        #     cur_equivilance_dist_sq = (true_row - cur_row)**2 + (true_col - cur_col)**2
        #     if cur_equivilance_dist_sq > equivilant_coord_dist_sq:
        #       continue
        #     # it's within the equivilance distance, check if it's actually close to a vertex
        #     near_vertex = False
        #     for i in range(3):
        #       candidate = verts[i, :].reshape(3, 1)
        #       if np.linalg.norm(candidate - vertex) < visibility_tolerance:
        #     # if so, retain the min one
        #
        # for equivilant_coord_dist
        #     return (
  return False


def read_pose(filename):
  transform = np.fromfile(filename, sep=' ').reshape(4, 4)
  R = transform[0:3, 0:3]
  t = transform[0:3, 3].reshape((3, 1))

  # wow, these random offsets aren't documented anywhere.
  # Apparently the TSDF is in millimeters, whereas the poses are in meters. Also there's a random 2x2x2 offset.
  t *= 1000
  t[0] += 2000
  t[1] += 2000
  t[2] += 2000
  return (R, t)


def find_descriptors(cache_dir, orig_filename):
  # frame-XXXXXX.pose.txt -> frame_XXXXX_keypoints_and_descriptors.bin
  id = orig_filename[6:12]
  descr_filename = 'frame_' + id + '_keypoints_and_descriptors.bin'
  descriptors = {}

  with open(os.path.join(cache_dir, descr_filename), 'rb') as f:
    num_descriptors = struct.unpack('=I', f.read(4))[0]
    descriptor_size = struct.unpack('=I', f.read(4))[0]
    # ignore this, just assume float32
    data_type = struct.unpack('=I', f.read(4))[0]

    # first descriptors
    all_descriptors = np.fromfile(f, dtype=np.float32, count=num_descriptors * descriptor_size).reshape(
      (num_descriptors, descriptor_size))

    # then keypoints

    for i in range(num_descriptors):
      col = int(round(struct.unpack('=f', f.read(4))[0]))
      row = int(round(struct.unpack('=f', f.read(4))[0]))
      # unused stuff
      size = struct.unpack('=f', f.read(4))[0]
      angle = struct.unpack('=f', f.read(4))[0]
      response = struct.unpack('=f', f.read(4))[0]
      octave = struct.unpack('=I', f.read(4))[0]
      class_id = struct.unpack('=I', f.read(4))[0]

      descriptors[(row, col)] = all_descriptors[i, :]
      # print 'descriptors[', row, ',', col, ']: ', descriptors[(row, col)]

  return descriptors

def getNameFromDescriptorCache(cache):
  last = cache.rfind('/', )
  penultimate = cache[:last-1].rfind('/')
  return cache[penultimate+1:last]

def main():
  print 'loading mesh...'
  mesh = trimesh.load_mesh('mesh.stl')
  print 'loaded!'

  # assumes full pipeline has already been run
  sequence_dir_1 = '/home/daniel/data/7scenes/fire/seq-03'
  sequence_dir_2 = '/home/daniel/data/7scenes/fire/seq-02'
  # SIFT
  # descriptor_cache_dir_1 = '/home/daniel/data/tmp/fire/seq-03'
  # descriptor_cache_dir_2 = '/home/daniel/data/tmp/fire/seq-02'
  # descriptor_cache_dir_1 = '/home/daniel/data/tmp/results-aligned-sparser/fire-orig/seq-03'
  # descriptor_cache_dir_2 = '/home/daniel/data/tmp/results-aligned-sparser/fire-orig/seq-02'
  # Learned
  descriptor_cache_dir_1 = '/home/daniel/data/tmp/fire-caffe/seq-03'
  descriptor_cache_dir_2 = '/home/daniel/data/tmp/fire-caffe/seq-02'
  # descriptor_cache_dir_1 = '/home/daniel/data/tmp/fire-caffe-all48/seq-03'
  # descriptor_cache_dir_2 = '/home/daniel/data/tmp/fire-caffe-all48/seq-02'
  # descriptor_cache_dir_1 = '/home/daniel/data/tmp/fire-caffe-all48-obs/seq-03'
  # descriptor_cache_dir_2 = '/home/daniel/data/tmp/fire-caffe-all48-obs/seq-02'

  name = getNameFromDescriptorCache(descriptor_cache_dir_1)

  num_times_first_nn = 0
  num_times_second_nn = 0
  num_times_not_nn = 0
  true_positive_dist_ratios = []
  false_positive_dist_ratios = []
  true_correspondence_dists = []
  false_correspondence_dists = []

  sorted_files_1 = sorted([f for f in sorted(os.listdir(sequence_dir_1)) if f.endswith('pose.txt')])
  sorted_files_2 = sorted([f for f in sorted(os.listdir(sequence_dir_2)) if f.endswith('pose.txt')])
  iteration = 0
  while True:
    iteration += 1
    print '\n\nIteration', iteration

    while True:
      # Pick two images
      # # For simplicity, keep them sort of close to each other
      # # (this only makes sense if both are in same dataset)
      # idx_offset = random.randint(5, 50)
      # first_idx = random.randint(0, len(sorted_files_1) - 1 - idx_offset)
      # second_idx = first_idx + idx_offset

      # just pick 2 at random. In the worst case, there's no overlap and we have to restart
      first_idx = random.randint(0, random.randint(0, len(sorted_files_1) - 1))
      second_idx = random.randint(0, random.randint(0, len(sorted_files_2) - 1))

      first_file = sorted_files_1[first_idx]
      second_file = sorted_files_2[second_idx]

      # check if they both have descriptors cached, otherwise restart
      first_descriptors = find_descriptors(descriptor_cache_dir_1, first_file)
      second_descriptors = find_descriptors(descriptor_cache_dir_2, second_file)

      # if not enough descriptors, try again
      # Usually this happens for two reasons
      # 1. One of the sequences is a database sequence, so non-keyframes have 0 descriptors (more likely)
      # 2. Tracking was very poor during a particular frame (less likely)
      if len(first_descriptors) < 100 or len(second_descriptors) < 100:
        print '\tretrying, not enough descriptors...'
        continue
      print 'First file has', len(first_descriptors), 'descriptors, and second file has', len(
        second_descriptors), 'descriptors.'

      first_pose = read_pose(os.path.join(sequence_dir_1, first_file))
      second_pose = read_pose(os.path.join(sequence_dir_2, second_file))

      visible_in_first = getSelfProjectingVertices(mesh, first_pose, coords_to_check=first_descriptors.keys())

      # plt.imshow(visible_in_first/4000)
      # plt.show()

      # TODO: we could also compare the viewing frustums and see if it's even possible to have co-observed points.
      # But I think that would only allow us to skip a few frames.

      # for each visible in first, project the vertex into the second image, then raycast from the second image
      # (to check for obscurance)
      true_correspondences = []
      for row in range(H):
        for col in range(W):
          if np.any(np.nonzero(visible_in_first[row, col, :])):
            visibility_coords = check_visible(mesh, second_pose, visible_in_first[row, col, :],
                                              coords_to_check=second_descriptors.keys())
            if visibility_coords:
              true_correspondences.append(((row, col), visibility_coords))
      print 'Num visible points in adjacent frame:', len(true_correspondences)
      # don't bother checking unless there's lots of covisible points
      if len(true_correspondences) > 5:
        break
      else:
        print '\tretrying, not enough points in adjacent frame\'s field of view...'

    # We now have a list of true positives
    # Furthermore, each frame usually has about 10K-20K descriptors
    # so there are 10K^2 possible matches, of which only 10K (and points nearby) are correct
    # We can evaluate that via brute force checking

    for ground_truth_pair in true_correspondences:
      first_coord = ground_truth_pair[0]
      second_coord = ground_truth_pair[1]
      first_descr = first_descriptors[first_coord]
      second_descr = second_descriptors[second_coord]

      # Ugh, we should call them 'image A' and 'image B', since we use 1st and 2nd here
      first_nn_coord = second_coord
      first_nn_dist = np.linalg.norm(first_descr - second_descr)
      second_nn_coord = second_coord
      second_nn_dist = first_nn_dist
      for alternative_second_coord in second_descriptors.keys():
        dist = np.linalg.norm(second_descriptors[alternative_second_coord] - first_descr)
        # Note: for the training datasets, this distance can be 0 if descriptors have been merged between images.
        if dist == 0.0:
          print 'Distance is zero for some reason.'
          print 'first descr: ', first_descr
          print 'second descr: ', second_descriptors[alternative_second_coord]
        true_correspondence_dists
        if dist < second_nn_dist:
          if dist < first_nn_dist:
            second_nn_dist = first_nn_dist
            second_nn_coord = first_nn_coord
            first_nn_dist = dist
            first_nn_coord = alternative_second_coord
          else:
            second_nn_dist = dist
            second_nn_coord = alternative_second_coord

        # check if it's more-or-less a true correspondence
        if np.linalg.norm(np.asarray(alternative_second_coord) - np.asarray(second_coord)) ** 2 < equivilant_coord_dist_sq:
          true_correspondence_dists.append(dist)
        else:
          false_correspondence_dists.append(dist)

      ratio = 10000
      if second_nn_dist != 0:
        ratio = first_nn_dist / second_nn_dist
        assert ratio <= 1.0
      # a frame is a true positive if its distance to second_coord is very small
      if np.linalg.norm(np.asarray(first_nn_coord) - np.asarray(second_coord)) ** 2 < equivilant_coord_dist_sq:
        num_times_first_nn += 1
        # it's a 1st nearest neighbor. Record the distance ratio for posterity
        true_positive_dist_ratios.append(ratio)
      else:
        # a false positive is the 1st nearest neighbor. boo!
        false_positive_dist_ratios.append(ratio)
        if np.linalg.norm(np.asarray(second_nn_coord) - np.asarray(second_coord)) ** 2 < equivilant_coord_dist_sq:
          num_times_second_nn += 1
        else:
          num_times_not_nn += 1

    print '============================='
    print '===== CUMULATIVE TOTALS ====='
    print '============================='
    print 'num_times_first_nn:', num_times_first_nn
    print 'num_times_second_nn:', num_times_second_nn
    print 'num_times_not_nn:', num_times_not_nn
    print '============================='

    if iteration % 20 == 0:
      num_bins = 100
      tp_pdf, bin_edges = np.histogram(true_positive_dist_ratios, bins=num_bins, range=[0.0, 1.0], density=True)
      fp_pdf, _ = np.histogram(false_positive_dist_ratios, bins=num_bins, range=[0.0, 1.0], density=True)
      bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
      plt.plot(bins, tp_pdf, 'g', label='True Positive PDF')
      plt.plot(bins, fp_pdf, 'r', label='False Positive PDF')
      plt.legend()
      plt.title('TP-FP for ' + name)
      plt.show()

      true_dist_pdf, bin_edges = np.histogram(true_correspondence_dists, bins=100, density=False)
      bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
      plt.plot(bins, true_dist_pdf, 'g', label='True Distance PDF')
      # use way more bins here, since we have more samples
      false_dist_pdf, bin_edges = np.histogram(false_correspondence_dists, bins=10000, density=False)
      bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
      plt.plot(bins, false_dist_pdf, 'r', label='False Distance PDF')
      plt.legend()
      plt.title('Distances for ' + name)
      plt.show()


if __name__ == "__main__":
  main()
