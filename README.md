# msc-thesis

Code for my thesis about relocalization for semi-dense visual odometry.

### Installation

This is tested on ubuntu 16.04. You will need a few tools:

cmake, eigen3, boost, libzip, pcl, libproj

sudo apt-get install cmake libeigen3-dev libboost-all-dev libzip-dev libpcl-dev libproj-dev

you'll also need opencv 3.0+. You can get it with ros, or install it from source, or apparently there's also a package called libopencv-dev, which I haven't checked out, but I think it might be an old version.

you will also needs DSO (see its installation instructions), with Pangolin.

For now, after compiling dso, you'll need to open the CMakeLists.txt file and adjust the line

set(DSO_PATH "/home/daniel/git/dso")

to point to your installation.

TODO: other things you'll need: my forks of DSO, geometric_burstiness, fastcluster, and fastann
