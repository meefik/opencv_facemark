#!/bin/sh

cmake "../opencv" \
 -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_opencv_face=ON \
 -DOPENCV_EXTRA_MODULES_PATH="../opencv_contrib/modules"

make -j$(grep -c ^processor /proc/cpuinfo)
