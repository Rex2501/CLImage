//
//  IntegralImage.cpp
//  ImageRegistration
//
//  Created by Fabio Riccardi on 9/21/22.
//

#include "IntegralImage.hpp"

#define INTEGRAL_USE_BUFFERS false

namespace gls {

void clIntegral(gls::OpenCLContext* glsContext, const gls::image<float>& img, const gls::cl_image_buffer_2d<float>* sum, const float bias) {
    static const int tileSize = 16;

    gls::size bufsize(((img.height + tileSize - 1) / tileSize) * tileSize, ((img.width + tileSize - 1) / tileSize) * tileSize);
    cl::Buffer tmpBuffer(CL_MEM_READ_WRITE, bufsize.width * bufsize.height * sizeof(float));

    // Load the shader source
    const auto program = glsContext->loadProgram("SURF");

#if INTEGRAL_USE_BUFFERS
    cl::Buffer imgBuffer(img.pixels().begin(), img.pixels().end(), true);

    // Bind the kernel parameters
    auto integral_sum_cols = cl::KernelFunctor<cl::Buffer,  // src_ptr
                                               int,         // src_width
                                               int,         // src_height
                                               cl::Buffer,  // buf_ptr
                                               int,         // buf_width
                                               float        // bias
                                               >(program, "integral_sum_cols");

    // Schedule the kernel on the GPU
    integral_sum_cols(cl::EnqueueArgs(cl::NDRange(img.width), cl::NDRange(tileSize)),
                      imgBuffer,
                      img.width,
                      img.height,
                      tmpBuffer,
                      bufsize.width,
                      bias);

    auto integral_sum_rows = cl::KernelFunctor<cl::Buffer,  // buf_ptr
                                               int,         // buf_width
                                               cl::Buffer,  // dst_ptr
                                               int,         // dst_width
                                               int          // dst_height
                                               >(program, "integral_sum_rows");

    // Schedule the kernel on the GPU
    integral_sum_rows(cl::EnqueueArgs(cl::NDRange(img.height), cl::NDRange(tileSize)),
                      tmpBuffer,
                      bufsize.width,
                      sum->getBuffer(),
                      sum->stride,
                      sum->height);
#else
    const auto image = gls::cl_image_2d<float>(glsContext->clContext(), img);

    // Bind the kernel parameters
    auto integral_sum_cols = cl::KernelFunctor<cl::Image2D, // src_ptr
                                               cl::Buffer,  // buf_ptr
                                               int,         // buf_width
                                               float        // bias
                                               >(program, "integral_sum_cols_image");

    // Schedule the kernel on the GPU
    integral_sum_cols(cl::EnqueueArgs(cl::NDRange(img.width), cl::NDRange(tileSize)),
                      image.getImage2D(),
                      tmpBuffer,
                      bufsize.width,
                      bias);

    // Bind the kernel parameters
    auto integral_sum_rows = cl::KernelFunctor<cl::Buffer,  // buf_ptr
    int,         // buf_width
    cl::Image2D  // dst_ptr
    >(program, "integral_sum_rows_image");

    // Schedule the kernel on the GPU
    integral_sum_rows(cl::EnqueueArgs(cl::NDRange(img.height), cl::NDRange(tileSize)),
                      tmpBuffer,
                      bufsize.width,
                      sum->getImage2D());
#endif
}

}
