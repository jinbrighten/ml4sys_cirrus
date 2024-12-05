cmake_minimum_required(VERSION 3.10)
project(KITTI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) 

add_library(utility STATIC src/utility.cpp)

add_library(knob STATIC src/knob.cpp)
target_link_libraries(knob utility)



add_executable(combination_sampling src/combination_sampling.cpp)
target_link_libraries (
        combination_sampling
        knob
        pthread
        ${VTK_LIBRARIES}
        ${PCL_LIBRARIES}
        jsoncpp_lib
)
