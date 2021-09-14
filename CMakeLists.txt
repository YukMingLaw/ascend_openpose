cmake_minimum_required(VERSION 3.5)

project(ACL_OPENPOSE)

set(CMAKE_CXX_STANDARD 11)
add_compile_options(-fPIC)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(OpenMP)

set(OpenCV_DIR /root/opencv4.4/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)

set(ACL_PATH /usr/local/Ascend/ascend-toolkit/latest)
INCLUDE_DIRECTORIES(
    ${ACL_PATH}/acllib/include
    ${OpenCV_INCLUDE_DIRS}
    AclProcess
    )

set(ACL_LIB_PATH ${ACL_PATH}/acllib/lib64/stub)
link_directories(${ACL_LIB_PATH})

add_executable(${PROJECT_NAME}
    main.cpp
    AclProcess/AclProcess.cpp
    AclProcess/ModelProcess.cpp
    AclProcess/openpose.cpp
    )

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX OpenMP::OpenMP_C ascendcl ${OpenCV_LIBS} pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -s stdc++)
