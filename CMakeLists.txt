cmake_minimum_required(VERSION 2.8.3)
project(loop_rift)

add_compile_options(-std=c++11)
add_compile_options(-std=gnu++11)

set(OpenCV_DIR '/opt/ros/kinetic/share/OpenCV-3.3.1-dev/')
set(CMAKE_BUILD_TYPE release)


find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PCL REQUIRED)

SET( EIGEN3_INCLUDE_DIRS "/usr/local/include/eigen3" )

include_directories(
  include
	${EIGEN3_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
)

add_executable(match_two_scan src/match_two_scan.cpp) 
target_link_libraries(match_two_scan ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} python2.7)
