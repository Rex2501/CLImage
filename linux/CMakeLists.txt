cmake_minimum_required(VERSION 3.16) 

project("climage")

# CL will throw an unknown error on Linux if it is not explicitly linked with pthread, see flags at the bottom
find_package(Threads REQUIRED)

add_library( libz STATIC IMPORTED )
set_target_properties( libz PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libz.a )
add_library( libpng STATIC IMPORTED )
set_target_properties( libpng PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libpng.a )
add_library( libjpeg STATIC IMPORTED )
set_target_properties( libjpeg PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libjpeg.a ) #TODO: Possibly use jpeg-turbo

# Self-compiled libtiff
add_library( libtiff STATIC IMPORTED )
set_target_properties( libtiff PROPERTIES IMPORTED_LOCATION /usr/local/lib/libtiff.a )
add_library( libtiffxx STATIC IMPORTED )
set_target_properties( libtiffxx PROPERTIES IMPORTED_LOCATION /usr/local/lib/libtiffxx.a )


set(CMAKE_CXX_STANDARD 20)
set(ROOT_DIR ${CMAKE_SOURCE_DIR}/../)

###
# The following library and executable will probably not be used - actual library for bindings is compiled in other repo.
###

###
### Rendering library 
###

add_library(
    climage
    STATIC
    ${ROOT_DIR}/src/gls_cl.cpp
    ${ROOT_DIR}/src/gls_cl_error.cpp
    ${ROOT_DIR}/src/gls_color_science.cpp
    ${ROOT_DIR}/src/gls_dng_lossless_jpeg.cpp
    ${ROOT_DIR}/src/gls_image_jpeg.cpp
    ${ROOT_DIR}/src/gls_image_png.cpp
    ${ROOT_DIR}/src/gls_image_tiff.cpp
    ${ROOT_DIR}/src/gls_logging.cpp
    ${ROOT_DIR}/src/gls_tiff_metadata.cpp
)

target_link_libraries(
    climage
    libjpeg
    libpng
    libz
    OpenCL
)

###
### Demo pipeline
###

# Copy blur kernel
file(COPY ${ROOT_DIR}/CLImage/app/src/main/cpp/OpenCL/blur.cl DESTINATION ${ROOT_DIR}/build/OpenCL)

add_executable(
        climage-test
        ${ROOT_DIR}/src/gls_cl.cpp
        ${ROOT_DIR}/src/gls_image_png.cpp
        ${ROOT_DIR}/src/gls_image_jpeg.cpp
        ${ROOT_DIR}/src/gls_image_tiff.cpp
        ${ROOT_DIR}/src/gls_tiff_metadata.cpp
        ${ROOT_DIR}/src/gls_dng_lossless_jpeg.cpp
        ${ROOT_DIR}/src/gls_cl_error.cpp
        ${ROOT_DIR}/src/gls_logging.cpp
        ${ROOT_DIR}/CLImage/app/src/main/cpp/cl_pipeline.cpp
        ${ROOT_DIR}/CLImage/app/src/main/cpp/main.cpp
)

target_include_directories( climage-test PRIVATE ${ROOT_DIR}/src )

target_link_libraries(
    climage-test
    libjpeg
    libpng
    libz
    libtiff
    libtiffxx
    OpenCL
)

# Setting pthread flags to prevent silent OpenCL error, works for g++ and Clang
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Werror=return-type")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
