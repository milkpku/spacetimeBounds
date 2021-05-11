cmake_minimum_required(VERSION 3.11)

cmake_policy(SET CMP0078 NEW)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS"-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(EIGEN_INCLUDE_PATH "" CACHE PATH "Source directory to eigen")
if (EIGEN_INCLUDE_PATH STREQUAL "")
      message(FATAL_ERROR "No Eigen directory specified")
endif()
include_directories(${EIGEN_INCLUDE_PATH})


include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(utils)
add_subdirectory(convhull)
add_subdirectory(unittest)

project(KinematicCore)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
set(CMAKE_SWIG_FLAGS "")

set(PYTHON_INCLUDE_PATH "" CACHE PATH "Python include dir")
if (PYTHON_INCLUDE_PATH STREQUAL "")
      message(FATAL_ERROR "No python include path specified")
endif()
include_directories(${PYTHON_INCLUDE_PATH})


set_property(SOURCE KinematicCore.i PROPERTY CPLUSPLUS ON)
swig_add_library(${PROJECT_NAME} LANGUAGE python SOURCES KinematicCore.i KinematicCore.cpp)
swig_link_libraries(${PROJECT_NAME} utils convhull)

add_custom_target(copy_kinematic ALL
  COMMAND cp -f "${CMAKE_BINARY_DIR}/_KinematicCore.so" "${CMAKE_BINARY_DIR}/KinematicCore.py" "" "${CMAKE_SOURCE_DIR}"
)
add_dependencies(copy_kinematic ${PROJECT_NAME})
