cmake_minimum_required(VERSION 3.10)
project(param_env)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(SDL)
find_package(SDL_image)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  rosbag
  std_msgs
  geometry_msgs
  pcl_conversions
)

include_directories(
  SYSTEM
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES param_env
#  DEPENDS system_lib
)

add_executable (structure_map src/structure_map.cpp)

target_link_libraries(structure_map
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES})


if(SDL_FOUND AND SDL_IMAGE_FOUND)
  add_executable (read_grid_map src/read_grid_map.cpp)

  target_link_libraries(read_grid_map
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${SDL_LIBRARY}
    ${SDL_IMAGE_LIBRARIES})

  add_dependencies(read_grid_map ${catkin_EXPORTED_TARGETS})
else()
  message(WARNING "SDL or SDL_image not found; skipping read_grid_map. structure_map remains enabled.")
endif()

  
