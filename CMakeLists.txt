#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)

PROJECT(testDlib)

include("/home/nushh/Downloads/dlib-18.18/dlib/cmake")

# Tell CMake to compile a program.  We do this with the ADD_EXECUTABLE()
# statement which takes the name of the output executable and then a list of
# .cpp files to compile.  Here each example consists of only one .cpp file but
# in general you will make programs that const of many .cpp files.

find_package(OpenCV)
#if (OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(webcam webcam.cpp)
TARGET_LINK_LIBRARIES(webcam dlib ${OpenCV_LIBS} )
#endif()

