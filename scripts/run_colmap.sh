    

#name="redkitchen"
#images=()
## be sure to only use the train set!
## TODO: parse train set file
#images+=(/home/daniel/data/7scenes/redkitchen/seq-01)
#images+=(/home/daniel/data/7scenes/redkitchen/seq-02)
#images+=(/home/daniel/data/7scenes/redkitchen/seq-05)
#images+=(/home/daniel/data/7scenes/redkitchen/seq-07)
#images+=(/home/daniel/data/7scenes/redkitchen/seq-08)
#images+=(/home/daniel/data/7scenes/redkitchen/seq-11)
#images+=(/home/daniel/data/7scenes/redkitchen/seq-13)

#name="OldHospital"
#images=()
#images+=("/home/daniel/data/cambridge_landmarks/OldHospital/seq1")
#images+=("/home/daniel/data/cambridge_landmarks/OldHospital/seq2")
#images+=("/home/daniel/data/cambridge_landmarks/OldHospital/seq3")
#images+=("/home/daniel/data/cambridge_landmarks/OldHospital/seq5")
#images+=("/home/daniel/data/cambridge_landmarks/OldHospital/seq6")
#images+=("/home/daniel/data/cambridge_landmarks/OldHospital/seq7")
#images+=("/home/daniel/data/cambridge_landmarks/OldHospital/seq9")

#name=KingsCollege
#images=()
#images+=("/home/daniel/data/cambridge_landmarks/KingsCollege/seq1")
#images+=("/home/daniel/data/cambridge_landmarks/KingsCollege/seq4")
#images+=("/home/daniel/data/cambridge_landmarks/KingsCollege/seq5")
#images+=("/home/daniel/data/cambridge_landmarks/KingsCollege/seq6")
#images+=("/home/daniel/data/cambridge_landmarks/KingsCollege/seq8")

#name=GreatCourt
#images=()
#images+=("/home/daniel/data/cambridge_landmarks/GreatCourt/seq2")
#images+=("/home/daniel/data/cambridge_landmarks/GreatCourt/seq3")
#images+=("/home/daniel/data/cambridge_landmarks/GreatCourt/seq5")

#name=GreatCourt-rescaled
#images=()
#images+=("/home/daniel/data/cambridge_landmarks_rescaled/GreatCourt/seq2")
#images+=("/home/daniel/data/cambridge_landmarks_rescaled/GreatCourt/seq3")
#images+=("/home/daniel/data/cambridge_landmarks_rescaled/GreatCourt/seq5")

name=StMarysChurch-rescaled
images=()
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq1")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq2")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq4")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq6")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq7")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq8")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq9")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq10")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq11")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq12")
images+=("/home/daniel/data/cambridge_landmarks_rescaled/StMarysChurch/seq14")

# define the camera model here. Also give an initial guess for the camera params--this is necessary for some models (eg FOV).
# If your initial guess for the center of projection is good, you might want to disable refine_print_pt 
#camera=SIMPLE_PINHOLE
#default_camera_params="624,388,220"
#refine_princ_pt=1

#camera=PINHOLE
#default_camera_params="624,624,387.484,221.77"
#refine_princ_pt=0

#camera=FOV
#default_camera_params="624,624,387.484,221.77,1.0"
#refine_princ_pt=0

camera=OPENCV
default_camera_params="659.759,652.627,387.484,221.77,0.0850642,-0.109577,0.0005628,0.000189794"
refine_princ_pt=0


output_path=/home/daniel/colmap/${name}-${camera}
tmp_path=$output_path/tmp
tmp_img_path=$tmp_path/images
tmp_snap_path=$tmp_path/snapshots
colmap_path=/home/daniel/git/colmap/build/src/exe/
vocab_tree=/home/daniel/Downloads/vocab_tree-65536.bin

mkdir -p $output_path
mkdir -p $tmp_path
mkdir -p $tmp_img_path
mkdir -p $tmp_snap_path

# make a bunch of symlinks to all the pictures
for d in ${images[@]}; do
    for file in "${d}"/*; do
        # condition for 7scenes format:
        #if [[ $file == *.color.png ]]
        # cambridge:
        if [[ $file == *.png ]]
        then
            newname=$tmp_img_path/$(basename $d)-$(basename $file)
            ln -s $file $newname
        fi
    done
done


$colmap_path/database_creator --database_path $output_path/database.db
$colmap_path/feature_extractor --database_path $output_path/database.db --image_path $tmp_img_path --ImageReader.camera_model $camera --ImageReader.camera_params "$default_camera_params" --ImageReader.single_camera 1
# sequential matcher doesn't ignore breaks between videos, but it still filters
# the putative matched images by number of inliers, so this should be fine.
# Also note: overlap is performed over powers of 2 (ie 0th, 1st, 2nd, 4th, 
# 8th, 16th following images), so for a dataset of 1000 images per sequence, 9
# is already large enough
$colmap_path/sequential_matcher --database_path $output_path/database.db --SequentialMatching.overlap 8 --SequentialMatching.loop_detection 1 --SequentialMatching.loop_detection_period 1 --SequentialMatching.vocab_tree_path $vocab_tree

$colmap_path/mapper --database_path $output_path/database.db --image_path $tmp_img_path --Mapper.init_num_trials 500 --Mapper.ba_refine_focal_length 1 --Mapper.ba_refine_principal_point $refine_princ_pt --Mapper.ba_local_max_num_iterations 25 --Mapper.ba_global_max_num_iterations 50 --Mapper.ba_global_use_pba 0 --Mapper.snapshot_path=$tmp_snap_path --Mapper.snapshot_images_freq 100 --export_path $output_path

# TODO: check what gets written to output path. might want to have mapper write to a tmp directory, then have model_coverter write to the actual output path.
$colmap_path/model_converter --input_path $output_path/0/ --output_path $output_path --output_type 'TXT'


