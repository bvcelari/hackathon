Silliest of apps to test Dlib and OpenCV working together.

OpenCV
------

You need to have a built version of OpenCV 2.4 (tested on OpenCV 2.4.13, Ubuntu).

Set an environment variable OpenCV_DIR pointing to a directory containing OpenCVConfig.cmake

Dlib
----
Download (no need to build) Dlib and set an environment variable DLIB_DIR pointing to the directory containing dlib/cmake.

e.g. export DLIB_DIR=

Download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract it to the source directory of this project, CMake will copy it where it belongs.
