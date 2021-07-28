cmake_minimum_required (VERSION 2.8)
project (body_from_image)

file(GLOB_RECURSE SRC_CPP *.cpp)

add_executable(${PROJECT_NAME}
    ${SRC_CPP}
)

message(STATUS "CONAN_LIBS=${CONAN_LIBS}")
message(STATUS "C4A_LIBS=${C4A_LIBS}")
target_link_libraries(${PROJECT_NAME}
    ${CONAN_LIBS}
    ${BUILT_LIBS}
    ${Caffe_LINK}
)