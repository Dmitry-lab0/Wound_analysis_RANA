cmake_minimum_required(VERSION 3.26.4.0)
cmake_policy(SET CMP0091 OLD)
cmake_policy(SET CMP0141 OLD)
cmake_policy(SET CMP0126 OLD)
cmake_policy(SET CMP0128 OLD)
project(CMAKE_TRY_COMPILE C)
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")
set(CMAKE_VERBOSE_MAKEFILE 1)
set(CMAKE_C_FLAGS "/DWIN32 /D_WINDOWS /W3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMPILE_DEFINITIONS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXE_LINKER_FLAGS}")
include_directories(${INCLUDE_DIRECTORIES})
set(CMAKE_SUPPRESS_REGENERATION 1)
link_directories(${LINK_DIRECTORIES})
cmake_policy(SET CMP0065 OLD)
cmake_policy(SET CMP0083 OLD)
include("${CMAKE_ROOT}/Modules/Internal/HeaderpadWorkaround.cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "C:/Users/Дима/Documents/ML/Analysis_of_wound/site/labeling/SAM-webui-main_v2/SAM-webui-main/midas/ros/midas_cpp/out/build/x64-Debug/CMakeFiles/CMakeScratch/TryCompile-am7x1a")
add_executable(cmTC_b9e67 "C:/Program Files/Microsoft Visual Studio/2022/Preview/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/share/cmake-3.26/Modules/CMakeCCompilerABI.c")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/cmTC_b9e67_loc"
     CONTENT $<TARGET_FILE:cmTC_b9e67>)
target_link_libraries(cmTC_b9e67 ${LINK_LIBRARIES})
