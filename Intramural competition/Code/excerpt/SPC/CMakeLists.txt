cmake_minimum_required(VERSION 3.10)
PROJECT(SPC)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

   # Set the path to the Boost libraries
   set(Boost_USE_STATIC_LIBS OFF)
   set(Boost_USE_MULTITHREADED ON)
   set(Boost_USE_STATIC_RUNTIME OFF)

   find_package(Boost REQUIRED COMPONENTS system thread)
   find_package(Qt5 COMPONENTS SerialPort REQUIRED)

   if(Boost_FOUND)
       include_directories(${Boost_INCLUDE_DIRS})
       link_directories(${Boost_LIBRARY_DIRS})
   endif()

   # Add your other source files here
   add_executable(SPC SPC.cpp)

   # Link Boost libraries to your executable
   target_link_libraries(SPC ${Boost_LIBRARIES})

   target_link_libraries(SPC Qt5::SerialPort)


