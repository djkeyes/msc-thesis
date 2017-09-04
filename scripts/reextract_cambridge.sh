

# The cambridge dataset contains videos, and frames with attached GPS coordinates.
# However, the video frames are extracted at a very low framerate (around 1.8 frames per second), whereas the original
# videos are a full 30 frames per second. This makes it very difficult to do any visual odometry. This re-extracts the videos.
# The original data seems to contain every 14th frame, and this script extracts every frame, so every 14th frame extracted corresponds to a ground truth position.
# It's also possible that these videos are extracted at exactly 2fps, but they drop some frames. Not sure, the math doesn't quite add up.

# By the way, if you want to verify this for yourself, you can run something like
# ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 SEQUENCE_NAME.mp4
# to get the original frame count

cambridge_orig=/home/daniel/data/cambridge_landmarks
cambridge_reextracted=/home/daniel/data/cambridge_landmarks_hifps_rescaled

for scene in "$cambridge_orig"/*; do
    scenename=$(basename "${scene}")
    for video in "${scene}"/videos/*; do
        filename=$(basename "${video}")
        seqname="${filename%.*}"
        
        output_dir="${cambridge_reextracted}/${scenename}/${seqname}"
        echo "Extracting ${video} to ${output_dir}/"
        mkdir -p "${output_dir}"
        # don't want jpg? want to change the quality? consider changing the file extension or the qscale:v (ranges from 2=best to 32=worst for jpg)
        # want every Kth frame? change mod(n\,1) -> mod(n\,K)
        # want a different resolution? play with the scale.
        ffmpeg -loglevel panic -i $video -vf "select=not(mod(n\,1)),scale=768:432" -qscale:v 3 "${output_dir}/frame%5d.jpg"
    done
done

# Also note that some of the videos are unused in the original dataset, and in particular that Street has weird names:
# img -> seq1 + seq2
# img_east -> east_walk
# img_north -> north_walk + north_walk2
# img_south -> south_walk
# img_west -> west_walk
# and img exclusively seems to be extracted at closer to 3 frames per second (every 10th frame)

