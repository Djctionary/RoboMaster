cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/usr/share/doc/tensorrt-8.2.1.9/cpp/)
## find tensorrt
include(FindPackageHandleStandardArgs)
 
## 用户可以输入的TensorRT 搜索路径
set(TensorRT_ROOT
	""
	CACHE
	PATH
	"TensorRT root directory")
 
## 设置TensorRT 搜索路径
set(TensorRT_SEARCH_PATH
  /usr/include/x86_64-linux-gnu
  /usr/src/tensorrt
  /usr/lib/x86_64-linux-gnu
  ${TensorRT_ROOT}
)
 
## 设置需要搜索的TensorRT 依赖库
set(TensorRT_ALL_LIBS
  nvinfer
  nvinfer_plugin
  nvparsers
  nvonnxparser
)
 
## 提前设置后面需要用的变量
set(TensorRT_LIBS_LIST)
set(TensorRT_LIBRARIES)
 
## 搜索头文件的路径
find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${TensorRT_SEARCH_PATH}
)
 
## 利用头文件路径下的version文件来设置TensorRT的版本信息
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")
 
  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()
message("TensorRT version: ${TensorRT_VERSION_STRING}")
 
## 搜索sample code的路径
find_path(
  TensorRT_SAMPLE_DIR
  NAMES trtexec/trtexec.cpp
  PATHS ${TensorRT_SEARCH_PATH}
  PATH_SUFFIXES samples
)
 
## 依次搜索TensorRT依赖库
foreach(lib ${TensorRT_ALL_LIBS} )
  find_library(
    TensorRT_${lib}_LIBRARY
    NAMES ${lib}
    PATHS ${TensorRT_SEARCH_PATH}
  )
  ## 存储TensorRT的依赖库变量
  set(TensorRT_LIBS_VARS TensorRT_${lib}_LIBRARY ${TensorRT_LIBS_LIST})
  ## 也是TensorRT的依赖库，存成list，方便后面用foreach
  list(APPEND TensorRT_LIBS_LIST TensorRT_${lib}_LIBRARY)
endforeach()
 
## 调用cmake内置功能，设置基础变量如xxx_FOUND
find_package_handle_standard_args(TensorRT REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_SAMPLE_DIR ${TensorRT_LIBS_VARS})
 
if(TensorRT_FOUND)
  ## 设置Tensor_LIBRARIES变量
  foreach(lib ${TensorRT_LIBS_LIST} )
    list(APPEND TensorRT_LIBRARIES ${${lib}})
  endforeach()
  message("Found TensorRT: ${TensorRT_INCLUDE_DIR} ${TensorRT_LIBRARIES} ${TensorRT_SAMPLE_DIR}")
  message("TensorRT version: ${TensorRT_VERSION_STRING}")
endif()
# target_link_libraries(yolov5 PRIVATE ${TensorRT_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

#Opencv
set(OpenCV_DIR ~/opencv-4.5.4/build)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5_det yolov5_det.cpp ${SRCS})
target_link_libraries(yolov5_det nvinfer)
target_link_libraries(yolov5_det cudart)
target_link_libraries(yolov5_det myplugins)
target_link_libraries(yolov5_det ${OpenCV_LIBS})

