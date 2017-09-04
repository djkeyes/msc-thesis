
# just rescales the cambridge dataset, no re-parsing of the original videos

cambridge_orig=/home/daniel/data/cambridge_landmarks
cambridge_rescaled=/home/daniel/data/cambridge_landmarks_rescaled

for scene in "$cambridge_orig"/*; do
    scenename=$(basename "${scene}")
    for sequence in "${scene}"/*/; do
        
        seqname=$(basename "${sequence}")
        output_dir="${cambridge_rescaled}/${scenename}/${seqname}"
        echo "Resizing images in ${seqname} to ${output_dir}/"
        mkdir -p "${output_dir}"
        for image in "${sequence}"/*.png; do
            imgname=$(basename "${image}")
            ffmpeg -loglevel panic -i $image -vf "scale=768:432" "${output_dir}/${imgname}"
        done
    done
done

