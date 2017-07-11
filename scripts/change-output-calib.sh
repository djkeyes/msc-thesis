
# Invoke this from the root directory of the tum dataset. The tum dataset uses
# 2 cameras, and undistorts them to 2 different pinhole models. This script
# changes the undistorted result to be the same pinhole calibration for both
# cameras, for convenience.

for d in ./*; do sed -i '3s/.*/0.43334375 0.6070875 0.488646875 0.500577083 0'/ $d/camera.txt; done
