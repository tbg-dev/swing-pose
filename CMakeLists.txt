cmake_minimum_required(VERSION 3.15.0)

project (pose)

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------
option(test "Build all tests." OFF)

# -----------------------------------------------------------------------------
# Common build setup
# -----------------------------------------------------------------------------
set(PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR})
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/build/cmake
)
include(BuildCommon)

list(REMOVE_ITEM CONAN_LIBS caffe caffe-d)
file(GLOB_RECURSE Caffe_LIBS ${CONAN_LIB_DIRS_CAFFE}/*caffe*)
if (BUILD_SHARED_LIBS)
    set(Caffe_LINK "${Caffe_LIBS}")
else()
    # If linking static, need to force_load caffe
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
        set(Caffe_LINK "-Wl,-force_load,${Caffe_LIBS}")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(Caffe_LINK -Wl,--whole-archive ${CONAN_LIBS_CAFFE} -Wl,--no-whole-archive)
    else()
        set(Caffe_LINK "${Caffe_LIBS}")
    endif()
endif()
message(STATUS "Caffe_LINK=${Caffe_LINK}")

# -----------------------------------------------------------------------------
# Link models
# -----------------------------------------------------------------------------
file(CREATE_LINK ${CONAN_OPENPOSE_ROOT}/models ${PROJECT_ROOT}/models SYMBOLIC)

# -----------------------------------------------------------------------------
# Add libraries
# -----------------------------------------------------------------------------
add_c4a_library(pose)

# -----------------------------------------------------------------------------
# Add executables/tests
# -----------------------------------------------------------------------------
add_subdirectory(test/body_from_image)