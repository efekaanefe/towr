cmake_minimum_required(VERSION 3.10)
project(go2_towr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # lets clangd find Eigen/TOWR headers via build/
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) # Debug builds of TOWR are very slow
endif()

find_package(towr REQUIRED)  # installed by `sudo make install` -> /usr/local/share/towr/cmake
find_package(ifopt REQUIRED)

add_executable(go2_motion go2_motion.cc)
target_link_libraries(go2_motion PRIVATE towr::towr ifopt::ifopt_ipopt)
