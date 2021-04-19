cmake_minimum_required(VERSION 3.11)

project(unittests)
enable_testing()
set(GTEST_ROOT ${VC_ENV_DIR}/lib64/gtest)
set(GTEST_LIBRARIES /usr/lib/libgtest.so
                    /usr/lib/libgtest_main.so)
file(GLOB test_list *.cpp)
add_executable(${PROJECT_NAME} ${test_list})
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES})
target_link_libraries(${PROJECT_NAME} utils)
target_link_libraries(${PROJECT_NAME} convhull)

add_custom_target(copy_test_data ALL
  COMMAND cp -f
  "${CMAKE_SOURCE_DIR}/unittest/atlas.txt"
  "${CMAKE_SOURCE_DIR}/unittest/atlas_one_joint_pos.txt"
  "${CMAKE_SOURCE_DIR}/unittest/atlas_one_link_pos.txt"
  "${CMAKE_SOURCE_DIR}/unittest/atlas_rest_joint_pos.txt"
  "${CMAKE_SOURCE_DIR}/unittest/atlas_rest_link_pos.txt"
  "${CMAKE_SOURCE_DIR}/unittest/humanoid3d.txt"
  "${CMAKE_SOURCE_DIR}/unittest/humanoid3d_pos_zero.txt"
  "${CMAKE_SOURCE_DIR}/unittest/humanoid3d_zero_pose.txt"
  ""
  "${CMAKE_BINARY_DIR}/unittest"
)
add_dependencies(copy_test_data ${PROJECT_NAME})
