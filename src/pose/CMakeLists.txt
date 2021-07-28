project (tbg_pose)

file(GLOB_RECURSE SRC_CPP *.cpp)

add_library(${PROJECT_NAME}
    ${SRC_CPP}
)

target_link_libraries(${PROJECT_NAME}
    ${CONAN_LIBS}
)