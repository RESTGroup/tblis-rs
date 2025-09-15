# Please note that options (`TBLIS_SRC`, `CINT_VER`, etc.) will be always be overrided
# in build.rs building script. However, this CMakeLists.txt is also valid for standalone
# building.

cmake_minimum_required(VERSION 3.15)
project(dep-tblis)

include(ExternalProject)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

if(TBLIS_SRC)
    message(STATUS "User specified directory of source code (tblis): ${TBLIS_SRC}")
else()
    set(TBLIS_SRC https://github.com/MatthewsResearchGroup/tblis.git)
    message(STATUS "Download source code (tblis): ${TBLIS_SRC}")
endif()

ExternalProject_Add(tblis
    URL ${TBLIS_SRC}
    PREFIX ${PROJECT_BINARY_DIR}/deps
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=RELEASE
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_INSTALL_PREFIX:PATH=${PROJECT_BINARY_DIR}/deps
)

install(DIRECTORY ${PROJECT_BINARY_DIR}/deps/${CMAKE_INSTALL_LIBDIR} DESTINATION ${CMAKE_INSTALL_PREFIX})
