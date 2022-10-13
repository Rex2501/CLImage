cmake_minimum_required(VERSION 3.16) 

project("climage")

add_library( libz STATIC IMPORTED )
set_target_properties( libz PROPERTIES IMPORTED_LOCATION /usr/lib32/libz.a )
add_library( libpng STATIC IMPORTED )
set_target_properties( libpng PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libpng.a )
add_library( libjpg STATIC IMPORTED )
set_target_properties( libjpg PROPERTIES IMPORTED_LOCATION /usr/lib32/libjpeg.a ) #TODO: Possibly use jpeg-turbo
add_library( libtiff STATIC IMPORTED )
set_target_properties( libpng PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libtiff.a )

set (CMAKE_CXX_STANDARD 20)

add_library(
    climage
    STATIC
    src/gls_cl.cpp
    src/gls_cl_error.cpp
    src/gls_color_science.cpp
    src/gls_dng_lossless_jpeg.cpp
    src/gls_image_jpeg.cpp
    src/gls_image_png.cpp
    src/gls_image_tiff.cpp
    src/gls_logging.cpp
    src/gls_tiff_metadata.cpp
)

target_link_libraries(
    climage
    libjpg
    libpng
    libz
    cl
)
