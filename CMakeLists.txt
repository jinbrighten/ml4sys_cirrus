cmake_minimum_required(VERSION 3.0.0)
project(KITTI)

set(CMAKE_CXX_STANDARD 17)
find_package( PCL REQUIRED )
include_directories( ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
# add_compile_definitions(EIGEN_DONT_VECTORIZE)

find_package(jsoncpp REQUIRED)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) 

add_library(utility STATIC src/utility.cpp)

add_library(knob STATIC src/knob.cpp)
target_link_libraries(knob utility)



add_executable(knob_combination src/knob_combination.cpp)
target_link_libraries (
        knob_combination
        knob
        pthread
        ${VTK_LIBRARIES}
        ${PCL_LIBRARIES}
        jsoncpp_lib
)
add_executable(knob_multi_resolution src/knob_multi_resolution.cpp)
target_link_libraries (
        knob_multi_resolution
        knob
        pthread
        ${VTK_LIBRARIES}
        ${PCL_LIBRARIES}
        jsoncpp_lib
)

