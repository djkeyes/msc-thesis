from collections import OrderedDict
import math
import os

import numpy as np
import cv2


# given an original cambridge landmarks dataset, and one reextracted using ffmpeg, find which frames in the original
# correspond to which frames in the new samples.


def image_loader(directory, starting_file = None):
  files = sorted(os.listdir(directory))
  for i in range(len(files)):
    file = files[i]

    # skip until we reach the desired start
    # tbh we could speed this up. oh well.
    if starting_file is not None:
      if file == starting_file:
        starting_file = None
      else:
        continue
    yield file, cv2.imread(os.path.join(directory, file))

def compute_ssd(a, b):
  if not a.shape == b.shape:
    # this assumes b has been rescaled down from a
    # downscale a, so the final result is cheaper to compute (and we don't have to worry about interpolating)
    a = cv2.resize(a, (b.shape[1], b.shape[0]))
  stereo = np.hstack((a, b))
  # cv2.imshow('', stereo)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  ssd = np.sum((a[:,:,0:3] - b[:,:,0:3])**2)
  return ssd / np.product(a.shape)

def process(orig_path, hifps_path, gt_file):
  sequences = set()
  orig_files_to_data = {}
  hifps_files_to_data = OrderedDict()

  with open(os.path.join(orig_path, gt_file)) as f:
    # first 3 lines are boilerplate
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()

    for line in f:
      split = line.split(' ', 1)
      filename = split[0]
      data = split[1]
      orig_files_to_data[filename] = data
      seq = filename[:filename.find('/')]
      sequences.add(seq)
  for sequence_dir in sorted(sequences):
    print('matching frames in ', sequence_dir)
    orig_seq = os.path.join(orig_path, sequence_dir)
    hifps_seq = os.path.join(hifps_path, sequence_dir)
    orig_loader = image_loader(orig_seq)
    hifps_loader = image_loader(hifps_seq)
    filename_orig, img_orig = next(orig_loader)
    filename_hifps, img_hifps = next(hifps_loader)
    frames_since_found = 0
    threshold = 20
    exp_avg_notmatched = 70
    exp_avg_matched = 7
    alpha = 0.8
    last_matched_hifps = filename_hifps
    while True:
      try:
        # increment cur_hifps until the image distance (here SSD) is sufficiently low
        dist = compute_ssd(img_orig, img_hifps)
        # print('dist: ', dist, 'threshold=', threshold, '[', exp_avg_matched, exp_avg_notmatched, ']')
        if dist < threshold:
          # filename_orig <--> filename_hifps

          hifps_parent_name = os.path.join(sequence_dir, filename_hifps)
          orig_parent_name = os.path.join(sequence_dir, filename_orig)
          # check for dropped frames
          if orig_parent_name in orig_files_to_data:
            hifps_files_to_data[hifps_parent_name] = orig_files_to_data[orig_parent_name]
          filename_orig, img_orig = next(orig_loader)
          filename_hifps, img_hifps = next(hifps_loader)
          last_matched_hifps = filename_hifps
          # print('frames since last match: ', frames_since_found)
          frames_since_found = 0

          # update lower bound for adaptive threshold
          exp_avg_matched = (1 - alpha) * dist + alpha * exp_avg_matched
        else:
          filename_hifps, img_hifps = next(hifps_loader)

          exp_avg_notmatched = (1 - alpha) * dist + alpha * exp_avg_notmatched

        threshold = (3*exp_avg_matched + exp_avg_notmatched) / 4
        frames_since_found += 1

        if frames_since_found > 60:
          # something's gone wrong
          # find the minimum in the last window
          print('LOST MATCHED FRAME!')
          print('past thresholds:', threshold, '[', exp_avg_matched, exp_avg_notmatched, ']')
          hifps_loader = image_loader(hifps_seq, last_matched_hifps)
          min_dist = math.inf
          total_dist = 0
          print('choosing min in past window:')
          for i in range(60):
            filename_hifps, img_hifps = next(hifps_loader)
            dist = compute_ssd(img_orig, img_hifps)
            print(dist)
            total_dist += dist
            if dist < min_dist:
              min_dist = dist
              best_filename_hifps = filename_hifps

          hifps_parent_name = os.path.join(sequence_dir, best_filename_hifps)
          orig_parent_name = os.path.join(sequence_dir, filename_orig)
          # check for dropped frames
          if orig_parent_name in orig_files_to_data:
            hifps_files_to_data[hifps_parent_name] = orig_files_to_data[orig_parent_name]

          exp_avg_notmatched = (total_dist - min_dist)/59 # or maybe this should be the 2nd smallest value
          exp_avg_matched = min_dist
          threshold = (3*exp_avg_matched + exp_avg_notmatched) / 4
          hifps_loader = image_loader(hifps_seq, best_filename_hifps)
          filename_orig, img_orig = next(orig_loader)
          filename_hifps, img_hifps = next(hifps_loader)
          frames_since_found = 0

          print('new thresholds:', threshold, '[', exp_avg_matched, exp_avg_notmatched, ']')

      except StopIteration:
        # why is this the idiomatic way to do this?
        break
  with open(os.path.join(hifps_path, gt_file), 'w') as f:
    f.write(line1)
    f.write(line2)
    f.write(line3)
    for filename, data in hifps_files_to_data.items():
      f.write(filename + ' ' + data)


def main():
  orig = '/home/daniel/data/cambridge_landmarks'
  hifps = '/home/daniel/data/cambridge_landmarks_hifps_rescaled'

  for directory in sorted(os.listdir(orig)):
    print('processing ', directory)
    if directory == 'GreatCourt':
      continue
    orig_path = os.path.join(orig, directory)
    hifps_path = os.path.join(hifps, directory)
    process(orig_path, hifps_path, 'dataset_test.txt')
    process(orig_path, hifps_path, 'dataset_train.txt')


if __name__ == "__main__":
  main()
