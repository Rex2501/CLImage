// Copyright (c) 2021-2022 Glass Imaging Inc.
// Author: Fabio Riccardi <fabio@glass-imaging.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "SURF.hpp"

#include <float.h>

#include <cmath>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "gls_cl_image.hpp"

#include "feature2d.hpp"

#include "ThreadPool.hpp"

#define USE_OPENCL true
#define USE_OPENCL_INTEGRAL true

template<>
struct std::hash<gls::size> {
    std::size_t operator()(gls::size const& r) const noexcept {
        return r.width ^ r.height;
    }
};

namespace surf {

static const int   SURF_ORI_SEARCH_INC = 5;
static const float SURF_ORI_SIGMA      = 2.5f;
static const float SURF_DESC_SIGMA     = 3.3f;

// Bias value used to optimize the dynamic range of the integral image
static const float SURF_INTEGRAL_BIAS  = 255;

// Wavelet size at first layer of first octave.
static const int SURF_HAAR_SIZE0 = 9;

// Wavelet size increment between layers. This should be an even number,
// such that the wavelet sizes in an octave are either all even or all odd.
// This ensures that when looking for the neighbours of a sample, the layers
// above and below are aligned correctly.
static const int SURF_HAAR_SIZE_INC = 6;

struct SurfHF {
    gls::point p0, p1, p2, p3;
    float w;

    SurfHF() : p0({0, 0}), p1({0, 0}), p2({0, 0}), p3({0, 0}), w(0) {}
};

template <size_t N>
float calcHaarPattern(const gls::image<float>& sum, const gls::point& p, const std::array<SurfHF, N>& f) {
    float d = 0;
    for (int k = 0; k < N; k++) {
        const auto& fk = f[k];

        d += SURF_INTEGRAL_BIAS * (sum[p.y + fk.p0.y][p.x + fk.p0.x] +
                                   sum[p.y + fk.p3.y][p.x + fk.p3.x] -
                                   sum[p.y + fk.p1.y][p.x + fk.p1.x] -
                                   sum[p.y + fk.p2.y][p.x + fk.p2.x]) * fk.w;
    }
    return d;
}

template <size_t N>
void resizeHaarPattern(const int src[][5], std::array<SurfHF, N>* dst, int oldSize, int newSize, int widthStep) {
    float ratio = (float) newSize / oldSize;
    for (int k = 0; k < N; k++) {
        int dx1 = (int)lrint(ratio * src[k][0]);
        int dy1 = (int)lrint(ratio * src[k][1]);
        int dx2 = (int)lrint(ratio * src[k][2]);
        int dy2 = (int)lrint(ratio * src[k][3]);
        (*dst)[k].p0 = { dx1, dy1 };
        (*dst)[k].p1 = { dx1, dy2 };
        (*dst)[k].p2 = { dx2, dy1 };
        (*dst)[k].p3 = { dx2, dy2 };
        (*dst)[k].w = src[k][4] / ((float)(dx2 - dx1) * (dy2 - dy1));
    }
}

struct clSurfHF {
    cl_int2 p0, p1, p2, p3;
    cl_float w;

    clSurfHF() {}

    clSurfHF(const SurfHF& cpp) :
        p0({cpp.p0.x, cpp.p0.y}),
        p1({cpp.p1.x, cpp.p1.y}),
        p2({cpp.p2.x, cpp.p2.y}),
        p3({cpp.p3.x, cpp.p3.y}),
        w(cpp.w) {}
};

struct SurfClContext {
    gls::OpenCLContext* glsContext = nullptr;

    std::mutex clMutex;
    std::mutex findMaximaInLayerMutex;

    std::unordered_map<gls::size, gls::cl_image_2d<float>*> sumMemory;
    std::unordered_map<gls::size, gls::cl_image_2d<float>*> detMemory;
    std::unordered_map<gls::size, gls::cl_image_2d<float>*> traceMemory;

    cl::Buffer DxBuffer;
    cl::Buffer DyBuffer;
    cl::Buffer DxyBuffer;

    cl::Buffer keyPointsBuffer;

    SurfClContext(gls::OpenCLContext* cLContext) {
        glsContext = cLContext;
    }
};

void clCalcDetAndTrace(SurfClContext *ctx,
                       const gls::cl_image_2d<float>& sumImage,
                       gls::cl_image_2d<float>* detImage,
                       gls::cl_image_2d<float>* traceImage,
                       const gls::rectangle& margin_crop,
                       const int sampleStep,
                       const std::array<SurfHF, 3>& Dx,
                       const std::array<SurfHF, 3>& Dy,
                       const std::array<SurfHF, 4>& Dxy) {
    // Load the shader source
    const auto program = ctx->glsContext->loadProgram("SURF");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // sumImage
                                    cl::Image2D,  // detImage
                                    cl::Image2D,  // traceImage
                                    cl_int2,      // margin
                                    int,          // sampleStep
                                    cl::Buffer,   // Dx
                                    cl::Buffer,   // Dy
                                    cl::Buffer    // Dxy
                                    >(program, "calcDetAndTrace");

    const std::array<clSurfHF, 3> clDx = { Dx[0], Dx[1], Dx[2] };
    const std::array<clSurfHF, 3> clDy = { Dy[0], Dy[1], Dy[2] };
    const std::array<clSurfHF, 4> clDxy = { Dxy[0], Dxy[1], Dxy[2], Dxy[3] };

    if (ctx->DxBuffer.get() == 0) {
        ctx->DxBuffer = cl::Buffer(CL_MEM_READ_ONLY, sizeof clDx);
        ctx->DyBuffer = cl::Buffer(CL_MEM_READ_ONLY, sizeof clDy);
        ctx->DxyBuffer = cl::Buffer(CL_MEM_READ_ONLY, sizeof clDxy);
    }
    cl::copy(clDx.begin(), clDx.end(), ctx->DxBuffer);
    cl::copy(clDy.begin(), clDy.end(), ctx->DyBuffer);
    cl::copy(clDxy.begin(), clDxy.end(), ctx->DxyBuffer);

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(margin_crop.width, margin_crop.height),
           sumImage.getImage2D(), detImage->getImage2D(), traceImage->getImage2D(),
           { margin_crop.x, margin_crop.y }, sampleStep, ctx->DxBuffer, ctx->DyBuffer, ctx->DxyBuffer);
}

#define INTEGRAL_USE_BUFFERS true

void clIntegral(gls::OpenCLContext* glsContext, const gls::image<float>& img, gls::cl_image_buffer_2d<float>* sum) {
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
                                               int          // buf_width
                                               >(program, "integral_sum_cols");

    // Schedule the kernel on the GPU
    integral_sum_cols(cl::EnqueueArgs(cl::NDRange(img.width), cl::NDRange(tileSize)),
                      imgBuffer,
                      img.width,
                      img.height,
                      tmpBuffer,
                      bufsize.width);

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
                                               int          // buf_width
                                               >(program, "integral_sum_cols_image");

    // Schedule the kernel on the GPU
    integral_sum_cols(cl::EnqueueArgs(cl::NDRange(img.width), cl::NDRange(tileSize)),
                      image.getImage2D(),
                      tmpBuffer,
                      bufsize.width);

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

void calcDetAndTrace(const gls::image<float>& sum,
                     gls::image<float>* det,
                     gls::image<float>* trace,
                     int x, int y, int sampleStep,
                     const std::array<SurfHF, 3>& Dx,
                     const std::array<SurfHF, 3>& Dy,
                     const std::array<SurfHF, 4>& Dxy) {
    const gls::point p = { x * sampleStep, y * sampleStep };

    float dx = calcHaarPattern(sum, p, Dx);
    float dy = calcHaarPattern(sum, p, Dy);
    float dxy = calcHaarPattern(sum, p, Dxy);

    (*det)[y][x] = dx * dy - 0.81f * dxy * dxy;
    (*trace)[y][x] = dx + dy;
}

void calcLayerDetAndTrace(SurfClContext *ctx, const gls::cl_image_2d<float>& sum, int size, int sampleStep, gls::cl_image_2d<float>* det, gls::cl_image_2d<float>* trace) {
    const int NX = 3, NY = 3, NXY = 4;

    const int dx_s[NX][5] = {
        {0, 2, 3, 7, 1},
        {3, 2, 6, 7, -2},
        {6, 2, 9, 7, 1}
    };
    const int dy_s[NY][5] = {
        {2, 0, 7, 3, 1},
        {2, 3, 7, 6, -2},
        {2, 6, 7, 9, 1}
    };
    const int dxy_s[NXY][5] = {
        {1, 1, 4, 4, 1},
        {5, 1, 8, 4, -1},
        {1, 5, 4, 8, -1},
        {5, 5, 8, 8, 1}
    };
    std::array<SurfHF, NX> Dx;
    std::array<SurfHF, NY> Dy;
    std::array<SurfHF, NXY> Dxy;

    if (size > (sum.height - 1) || size > (sum.width - 1))
        return;

    resizeHaarPattern(dx_s, &Dx, 9, size, sum.width);
    resizeHaarPattern(dy_s, &Dy, 9, size, sum.width);
    resizeHaarPattern(dxy_s, &Dxy, 9, size, sum.width);

    /* The integral image 'sum' is one pixel bigger than the source image */
    int height = 1 + (sum.height - 1 - size) / sampleStep;
    int width = 1 + (sum.width - 1 - size) / sampleStep;

    /* Ignore pixels where some of the kernel is outside the image */
    int margin = (size / 2) / sampleStep;

    gls::rectangle margin_crop = {margin, margin, width, height};

#if USE_OPENCL
    clCalcDetAndTrace(ctx, sum, det, trace, margin_crop, sampleStep, Dx, Dy, Dxy);
#else
    gls::image<float> sumCpu = sum.mapImage();
    gls::image<float> detCpuFull = det->mapImage();
    gls::image<float> traceCpuFull = trace->mapImage();

    gls::image<float> detCpu = gls::image<float>(detCpuFull, margin_crop);
    gls::image<float> traceCpu = gls::image<float>(traceCpuFull, margin_crop);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            calcDetAndTrace(sumCpu, &detCpu, &traceCpu, x, y, sampleStep, Dx, Dy, Dxy);
        }
    }

    sum.unmapImage(sumCpu);
    det->unmapImage(detCpuFull);
    trace->unmapImage(traceCpuFull);
#endif
}

/*
 * Maxima location interpolation as described in "Invariant Features from
 * Interest Point Groups" by Matthew Brown and David Lowe. This is performed by
 * fitting a 3D quadratic to a set of neighbouring samples.
 *
 * The gradient vector and Hessian matrix at the initial keypoint location are
 * approximated using central differences. The linear system Ax = b is then
 * solved, where A is the Hessian, b is the negative gradient, and x is the
 * offset of the interpolated maxima coordinates from the initial estimate.
 * This is equivalent to an iteration of Netwon's optimisation algorithm.
 *
 * N9 contains the samples in the 3x3x3 neighbourhood of the maxima
 * dx is the sampling step in x
 * dy is the sampling step in y
 * ds is the sampling step in size
 * point contains the keypoint coordinates and scale to be modified
 *
 * Return value is 1 if interpolation was successful, 0 on failure.
 */

inline bool solve3x3(const gls::Matrix<3, 3>& A, const gls::Vector<3>& b, gls::Vector<3>* x) {
    float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (det != 0) {
        float invdet = 1.0f / det;
        (*x)[0] = invdet *
               (b[0]    * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (b[1]    * A[2][2] - A[1][2] * b[2]   ) +
                A[0][2] * (b[1]    * A[2][1] - A[1][1] * b[2]   ));

        (*x)[1] = invdet *
               (A[0][0] * (b[1]    * A[2][2] - A[1][2] * b[2]   ) -
                b[0]    * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * b[2]    - b[1]    * A[2][0]));

        (*x)[2] = invdet *
               (A[0][0] * (A[1][1] * b[2]    - b[1]    * A[2][1]) -
                A[0][1] * (A[1][0] * b[2]    - b[1]    * A[2][0]) +
                b[0]    * (A[1][0] * A[2][1] - A[1][1] * A[2][0]));

        return true;
    }
    return false;
}

static bool interpolateKeypoint(const std::array<gls::image<float>, 3>& N9, int dx, int dy, int ds, KeyPoint* kpt) {
    gls::Vector<3> B = {
        -(N9[1][ 0][ 1] - N9[1][ 0][-1]) / 2, // Negative 1st deriv with respect to x
        -(N9[1][ 1][ 0] - N9[1][-1][ 0]) / 2, // Negative 1st deriv with respect to y
        -(N9[2][ 0][ 0] - N9[0][ 0][ 0]) / 2  // Negative 1st deriv with respect to s
    };
    gls::Matrix<3, 3> A = {
        {  N9[1][ 0][-1] - 2 * N9[1][ 0][ 0] + N9[1][ 0][ 1],                           // 2nd deriv x, x
          (N9[1][ 1][ 1] -     N9[1][ 1][-1] - N9[1][-1][ 1] + N9[1][-1][-1]) / 4,      // 2nd deriv x, y
          (N9[2][ 0][ 1] -     N9[2][ 0][-1] - N9[0][ 0][ 1] + N9[0][ 0][-1]) / 4 },    // 2nd deriv x, s
        { (N9[1][ 1][ 1] -     N9[1][ 1][-1] - N9[1][-1][ 1] + N9[1][-1][-1]) / 4,      // 2nd deriv x, y
           N9[1][-1][ 0] - 2 * N9[1][ 0][ 0] + N9[1][ 1][ 0],                           // 2nd deriv y, y
          (N9[2][ 1][ 0] -     N9[2][-1][ 0] - N9[0][ 1][ 0] + N9[0][-1][ 0]) / 4 },    // 2nd deriv y, s
        { (N9[2][ 0][ 1] -     N9[2][ 0][-1] - N9[0][ 0][ 1] + N9[0][ 0][-1]) / 4,      // 2nd deriv x, s
          (N9[2][ 1][ 0] -     N9[2][-1][ 0] - N9[0][ 1][ 0] + N9[0][-1][ 0]) / 4,      // 2nd deriv y, s
           N9[0][ 0][ 0] - 2 * N9[1][ 0][ 0] + N9[2][ 0][ 0] }                          // 2nd deriv s, s
    };
    // gls::Vector<3> x = B * gls::inverse(A);
    // gls::Vector<3> x = surf::inverse(A) * B;
    gls::Vector<3> x;
    bool ok = solve3x3(A, B, &x);
    ok = ok && (x[0] != 0 || x[1] != 0 || x[2] != 0) &&
         std::abs(x[0]) <= 1 && std::abs(x[1]) <= 1 && std::abs(x[2]) <= 1;

    if (ok) {
        kpt->pt.x += x[0] * dx;
        kpt->pt.y += x[1] * dy;
        kpt->size = rint(kpt->size + x[2] * ds);
    }
    return ok;
}

void SURFBuildInvoker(SurfClContext *ctx, const gls::cl_image_2d<float>& sum, std::vector<int>& sizes, std::vector<int>& sampleSteps,
                      const std::vector<gls::cl_image_2d<float>*>& dets, const std::vector<gls::cl_image_2d<float>*> traces) {
    ThreadPool threadPool(8);

    int N = (int) sizes.size();
    std::cout << "enqueueing " << N << " calcLayerDetAndTrace" << std::endl;

    for (int i = 0; i < N; i++) {
        // std::cout << "calcLayerDetAndTrace " << i << " of " << N << " @ " << dets[i]->width << ", " << dets[i]->width << std::endl;
        // threadPool.enqueue([&, i]() {
            calcLayerDetAndTrace(ctx, sum, sizes[i], sampleSteps[i], dets[i], traces[i]);
        // });
    }
}

/*
 * Find the maxima in the determinant of the Hessian in a layer of the
 * scale-space pyramid
 */

typedef struct KeyPointMaxima {
    static const constexpr int MaxCount = 20000;
    int count;
    KeyPoint keyPoints[MaxCount];
} KeyPointMaxima;

void clFindMaximaInLayer(SurfClContext *ctx, const gls::cl_image_2d<float>& sumImage,
                         const std::array<const gls::cl_image_2d<float>*, 3>& dets,
                         const gls::cl_image_2d<float>& traceImage,
                         const std::array<int, 3>& sizes, std::vector<KeyPoint>* keypoints, int octave,
                         float hessianThreshold, int sampleStep) {
    // Load the shader source
    const auto program = ctx->glsContext->loadProgram("SURF");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // detImage0
                                    cl::Image2D,  // detImage1
                                    cl::Image2D,  // detImage2
                                    cl::Image2D,  // traceImage
                                    cl_int3,      // sizes
                                    cl::Buffer,   // keypoints
                                    int,          // margin
                                    int,          // octave
                                    float,        // hessianThreshold
                                    int           // sampleStep
                                    >(program, "findMaximaInLayer");


    if (ctx->keyPointsBuffer.get() == 0) {
        ctx->keyPointsBuffer = cl::Buffer(CL_MEM_READ_WRITE, sizeof(KeyPointMaxima));
    }

    // The integral image 'sum' is one pixel bigger than the source image
    const int layer_height = (sumImage.height - 1) / sampleStep;
    const int layer_width = (sumImage.width - 1) / sampleStep;

    // Ignore pixels without a 3x3x3 neighbourhood in the layer above
    const int margin = (sizes[2] / 2) / sampleStep + 1;

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(layer_width - 2 * margin, layer_height - 2 * margin),
           dets[0]->getImage2D(), dets[1]->getImage2D(), dets[2]->getImage2D(),
           traceImage.getImage2D(),
           { sizes[0], sizes[1], sizes[2] }, ctx->keyPointsBuffer,
           margin, octave, hessianThreshold, sampleStep);

    const auto keyPointMaxima = (KeyPointMaxima *) cl::enqueueMapBuffer(ctx->keyPointsBuffer, true, CL_MAP_READ, 0, sizeof(KeyPointMaxima));

    std::cout << "keyPointMaxima: " << keyPointMaxima->count << std::endl;

    std::span<KeyPoint> newElements(keyPointMaxima->keyPoints, std::min(keyPointMaxima->count, KeyPointMaxima::MaxCount));

    keypoints->insert(end(*keypoints), begin(newElements), end(newElements));

    // Reset count
    keyPointMaxima->count = 0;

    cl::enqueueUnmapMemObject(ctx->keyPointsBuffer, (void*)keyPointMaxima);
}

void findMaximaInLayer(SurfClContext *ctx, const gls::image<float>& sum,
                       const std::array<gls::image<float>*, 3>& dets, const gls::image<float>& trace,
                       const std::array<int, 3>& sizes, std::vector<KeyPoint>* keypoints, int octave,
                       float hessianThreshold, int sampleStep) {
    const int size = sizes[1];

    // The integral image 'sum' is one pixel bigger than the source image
    const int layer_height = (sum.height - 1) / sampleStep;
    const int layer_width = (sum.width - 1) / sampleStep;

    // Ignore pixels without a 3x3x3 neighbourhood in the layer above
    const int margin = (sizes[2] / 2) / sampleStep + 1;

    const gls::image<float>& det0 = *dets[0];
    const gls::image<float>& det1 = *dets[1];
    const gls::image<float>& det2 = *dets[2];

    int keyPointMaxima = 0;
    for (int y = margin; y < layer_height - margin; y++) {
        for (int x = margin; x < layer_width - margin; x++) {
            const float val0 = (*dets[1])[y][x];

            if (val0 > hessianThreshold) {
                /* Coordinates for the start of the wavelet in the sum image. There
                   is some integer division involved, so don't try to simplify this
                   (cancel out sampleStep) without checking the result is the same */
                int sum_y = sampleStep * (y - (size / 2) / sampleStep);
                int sum_x = sampleStep * (x - (size / 2) / sampleStep);

                /* The 3x3x3 neighbouring samples around the maxima.
                   The maxima is included at N9[1][0][0] */

                const std::array<gls::image<float>, 3> N9 = {
                    gls::image<float>(det0, {x, y, 1, 1}),
                    gls::image<float>(det1, {x, y, 1, 1}),
                    gls::image<float>(det2, {x, y, 1, 1}),
                };

                /* Non-maxima suppression. val0 is at N9[1][0][0] */
                if (val0 > N9[0][-1][-1] && val0 > N9[0][-1][0] && val0 > N9[0][-1][1] &&
                    val0 > N9[0][ 0][-1] && val0 > N9[0][ 0][0] && val0 > N9[0][ 0][1] &&
                    val0 > N9[0][ 1][-1] && val0 > N9[0][ 1][0] && val0 > N9[0][ 1][1] &&
                    val0 > N9[1][-1][-1] && val0 > N9[1][-1][0] && val0 > N9[1][-1][1] &&
                    val0 > N9[1][ 0][-1]                        && val0 > N9[1][ 0][1] &&
                    val0 > N9[1][ 1][-1] && val0 > N9[1][ 1][0] && val0 > N9[1][ 1][1] &&
                    val0 > N9[2][-1][-1] && val0 > N9[2][-1][0] && val0 > N9[2][-1][1] &&
                    val0 > N9[2][ 0][-1] && val0 > N9[2][ 0][0] && val0 > N9[2][ 0][1] &&
                    val0 > N9[2][ 1][-1] && val0 > N9[2][ 1][0] && val0 > N9[2][ 1][1])
                {
                    /* Calculate the wavelet center coordinates for the maxima */
                    float center_y = sum_y + (size - 1) * 0.5f;
                    float center_x = sum_x + (size - 1) * 0.5f;
                    KeyPoint kpt = {{center_x, center_y}, (float)sizes[1], -1, val0, octave, trace[y][x] > 0 };

                    /* Interpolate maxima location within the 3x3x3 neighbourhood  */
                    int ds = size - sizes[0];
                    int interp_ok = interpolateKeypoint(N9, sampleStep, sampleStep, ds, &kpt);

                    /* Sometimes the interpolation step gives a negative size etc. */
                    if (interp_ok) {
                        std::lock_guard<std::mutex> guard(ctx->findMaximaInLayerMutex);
                        keypoints->push_back(kpt);
                        keyPointMaxima++;
                    }
                }
            }
        }
    }
    std::cout << "keyPointMaxima: " << keyPointMaxima << std::endl;
}

// --- Large Matrix Inversion ---

// TODO: replace this with a proper solver

template <size_t N, size_t M, typename baseT>
void swap_rows(gls::Matrix<N, M, baseT>& m, size_t i, size_t j) {
    for (size_t column = 0; column < M; column++)
        std::swap(m[i][column], m[j][column]);
}

// Convert matrix to reduced row echelon form
template <size_t N, size_t M, typename baseT>
void rref(gls::Matrix<N, M, baseT>& m) {
    for (size_t row = 0, lead = 0; row < N && lead < M; ++row, ++lead) {
        size_t i = row;
        while (m[i][lead] == 0) {
            if (++i == N) {
                i = row;
                if (++lead == M)
                    return;
            }
        }
        swap_rows(m, i, row);
        if (m[row][lead] != 0) {
            baseT f = m[row][lead];
            for (size_t column = 0; column < M; ++column)
                m[row][column] /= f;
        }
        for (size_t j = 0; j < N; ++j) {
            if (j == row)
                continue;
            baseT f = m[j][lead];
            for (size_t column = 0; column < M; ++column)
                m[j][column] -= f * m[row][column];
        }
    }
}

template <size_t N, typename baseT>
gls::Matrix<N, N, baseT> inverse(const gls::Matrix<N, N, baseT>& m) {
    gls::Matrix<N, 2 * N, baseT> tmp;
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < N; ++column)
            tmp[row][column] = m[row][column];
        tmp[row][row + N] = 1;
    }
    rref(tmp);
    gls::Matrix<N, N, baseT> inv;
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < N; ++column)
            inv[row][column] = tmp[row][column + N];
    }
    return inv;
}

void SURFFindInvoker(SurfClContext *ctx, const gls::cl_image_2d<float>& sum, const std::vector<gls::cl_image_2d<float>*>& dets,
                     const std::vector<gls::cl_image_2d<float>*>& traces, const std::vector<int>& sizes,
                     const std::vector<int>& sampleSteps, const std::vector<int>& middleIndices,
                     std::vector<KeyPoint>* keypoints, int nOctaveLayers, float hessianThreshold) {
    ThreadPool threadPool(8);

    int M = (int) middleIndices.size();
    std::cout << "enqueueing " << M << " findMaximaInLayer" << std::endl;
    for (int i = 0; i < M; i++) {
        const int layer = middleIndices[i];
        const int octave = i / nOctaveLayers;

#if USE_OPENCL
        const std::array<const gls::cl_image_2d<float>*, 3> detImages = {
            dets[layer-1],
            dets[layer],
            dets[layer+1]
        };

        const auto traceImage = traces[layer];

        clFindMaximaInLayer(ctx, sum, detImages, *traceImage,
                            { sizes[layer-1], sizes[layer], sizes[layer+1] },
                            keypoints, octave, hessianThreshold, sampleSteps[layer]);
#else
        const auto sumImage = sum.mapImage();
        auto dets0 = dets[layer-1]->mapImage();
        auto dets1 = dets[layer]->mapImage();
        auto dets2 = dets[layer+1]->mapImage();

        auto traceImage = traces[layer]->mapImage();

        findMaximaInLayer(ctx, sumImage, { &dets0, &dets1, &dets2 },
                          traceImage, { sizes[layer-1], sizes[layer], sizes[layer+1] },
                          keypoints, octave, hessianThreshold, sampleSteps[layer]);

        sum.unmapImage(sumImage);
        dets[layer-1]->unmapImage(dets0);
        dets[layer]->unmapImage(dets1);
        dets[layer+1]->unmapImage(dets2);
        traces[layer]->unmapImage(traceImage);
#endif
    }
}

struct KeypointGreater {
    inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const {
        if (kp1.response > kp2.response) return true;
        if (kp1.response < kp2.response) return false;
        if (kp1.size > kp2.size) return true;
        if (kp1.size < kp2.size) return false;
        if (kp1.octave > kp2.octave) return true;
        if (kp1.octave < kp2.octave) return false;
        if (kp1.pt.y < kp2.pt.y) return false;
        if (kp1.pt.y > kp2.pt.y) return true;
        return kp1.pt.x < kp2.pt.x;
    }
};

void fastHessianDetector(SurfClContext *ctx, const gls::cl_image_2d<float>& sum, std::vector<KeyPoint>* keypoints, int nOctaves,
                         int nOctaveLayers, float hessianThreshold) {
    /* Sampling step along image x and y axes at first octave. This is doubled
       for each additional octave. WARNING: Increasing this improves speed,
       however keypoint extraction becomes unreliable. */
    int SAMPLE_SETEPO = 1;

    int nTotalLayers = (nOctaveLayers + 2) * nOctaves;
    int nMiddleLayers = nOctaveLayers * nOctaves;

    std::vector<gls::cl_image_2d<float>*> dets(nTotalLayers);
    std::vector<gls::cl_image_2d<float>*> traces(nTotalLayers);
    std::vector<int> sizes(nTotalLayers);
    std::vector<int> sampleSteps(nTotalLayers);
    std::vector<int> middleIndices(nMiddleLayers);

    keypoints->clear();

    // Allocate space and calculate properties of each layer
    int index = 0, middleIndex = 0, step = SAMPLE_SETEPO;

    // const auto sumImage = gls::cl_image_2d<float>(ctx->glsContext->clContext(), sum);

    for (int octave = 0; octave < nOctaves; octave++) {
        for (int layer = 0; layer < nOctaveLayers + 2; layer++) {
            /* The integral image sum is one pixel bigger than the source image*/
            dets[index] = new gls::cl_image_2d<float>(ctx->glsContext->clContext(), (sum.width - 1) / step, (sum.height - 1) / step);
            traces[index] = new gls::cl_image_2d<float>(ctx->glsContext->clContext(), (sum.width - 1) / step, (sum.height - 1) / step);
            sizes[index] = (SURF_HAAR_SIZE0 + SURF_HAAR_SIZE_INC * layer) << octave;
            sampleSteps[index] = step;

            if (0 < layer && layer <= nOctaveLayers) {
                middleIndices[middleIndex++] = index;
            }
            index++;
        }
        step *= 2;
    }

    // Calculate hessian determinant and trace samples in each layer
    SURFBuildInvoker(ctx, sum, sizes, sampleSteps, dets, traces);

    // Find maxima in the determinant of the hessian
    SURFFindInvoker(ctx, sum, dets, traces, sizes,
                    sampleSteps, middleIndices, keypoints,
                    nOctaveLayers, hessianThreshold);

    sort(keypoints->begin(), keypoints->end(), KeypointGreater());

    for (int i = 0; i < nTotalLayers; i++) {
        delete dets[i];
        delete traces[i];
    }
}

void SURF_detect(SurfClContext *ctx, const gls::image<float>& srcImg, const gls::cl_image_2d<float>& integralSum, std::vector<KeyPoint>* keypoints, float hessianThreshold) {
    int nOctaves = 4;
    int nOctaveLayers = 2;
    fastHessianDetector(ctx, integralSum, keypoints, nOctaves, nOctaveLayers, hessianThreshold);
}

std::vector<float> getGaussianKernel(int n, float sigma) {
    const int SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] = {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}};

    const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ? small_gaussian_tab[n >> 1] : nullptr;

    float sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
    float scale2X = -0.5 / (sigmaX * sigmaX);
    float sum = 0;

    std::vector<float> kernel(n);
    for (int i = 0; i < n; i++) {
        float x = i - (n - 1) * 0.5;
        float t = fixed_kernel ? (float)fixed_kernel[i] : std::exp(scale2X * x * x);
        kernel[i] = t;
        sum += kernel[i];
    }

    sum = 1. / sum;
    for (auto& v : kernel) {
        v *= sum;
    }

    return kernel;
}

void resizeVV(const gls::image<float>& src, gls::image<float>* dst, int interpolation) {
    // Note that src and dst represent square matrices

    float dsize = (float)src.height / dst->height;
    if (dst->height && dst->width) {
        for (int i = 0; i < dst->height; i++) {
            for (int j = 0; j < dst->width; j++) {
                int _i = (int)i * dsize;       // integer part
                float _idec = i * dsize - _i;  // fractional part
                int _j = (int)j * dsize;
                float _jdec = j * dsize - _j;
                if (_j >= 0 && _j < (src.height - 1) && _i >= 0 && _i < (src.height - 1)) {
                    // Bilinear interpolation
                    (*dst)[i][j] = (1 - _idec) * (1 - _jdec) * src[_i][_j] + _idec * (1 - _jdec) * src[_i + 1][_j] +
                                   _jdec * (1 - _idec) * src[_j][_j + 1] + _idec * _jdec * src[_i + 1][_j + 1];
                }
            }
        }
    }
}

struct SURFInvoker {
    enum { ORI_RADIUS = 6, ORI_WIN = 60, PATCH_SZ = 20 };

    // Simple bound for number of grid points in circle of radius ORI_RADIUS
    const int nOriSampleBound = (2 * ORI_RADIUS + 1) * (2 * ORI_RADIUS + 1);

    // Parameters
    const gls::image<float>& img;
    const gls::image<float>& sum;
    std::vector<KeyPoint>* keypoints;
    gls::image<float>* descriptors;

    // Pre-calculated values
    int nOriSamples;
    std::vector<Point2f> apt;
    std::vector<float> aptw;
    std::vector<float> DW;

    SURFInvoker(const gls::image<float>& _img, const gls::image<float>& _sum, std::vector<KeyPoint>* _keypoints, gls::image<float>* _descriptors) :
        img(_img), sum(_sum), keypoints(_keypoints), descriptors(_descriptors) {
        enum { ORI_RADIUS = 6, ORI_WIN = 60, PATCH_SZ = 20 };

        // Simple bound for number of grid points in circle of radius ORI_RADIUS
        const int nOriSampleBound = (2 * ORI_RADIUS + 1) * (2 * ORI_RADIUS + 1);

        // Allocate arrays
        apt.resize(nOriSampleBound);
        aptw.resize(nOriSampleBound);
        DW.resize(PATCH_SZ * PATCH_SZ);

        /* Coordinates and weights of samples used to calculate orientation */
        const auto G_ori = getGaussianKernel(2 * ORI_RADIUS + 1, SURF_ORI_SIGMA);
        nOriSamples = 0;
        for (int i = -ORI_RADIUS; i <= ORI_RADIUS; i++) {
            for (int j = -ORI_RADIUS; j <= ORI_RADIUS; j++) {
                if (i * i + j * j <= ORI_RADIUS * ORI_RADIUS) {
                    apt[nOriSamples] = Point2f(i, j);
                    aptw[nOriSamples++] = G_ori[i + ORI_RADIUS] * G_ori[j + ORI_RADIUS];
                }
            }
        }
        assert( nOriSamples <= nOriSampleBound );

        /* Gaussian used to weight descriptor samples */
        const auto G_desc = getGaussianKernel(PATCH_SZ, SURF_DESC_SIGMA);
        for (int i = 0; i < PATCH_SZ; i++) {
            for (int j = 0; j < PATCH_SZ; j++) DW[i * PATCH_SZ + j] = G_desc[i] * G_desc[j];
        }
    }

    void computeRange(int k1, int k2) {
        /* X and Y gradient wavelet data */
        const int NX = 2, NY = 2;
        const int dx_s[NX][5] = {{0, 0, 2, 4, -1}, {2, 0, 4, 4, 1}};
        const int dy_s[NY][5] = {{0, 0, 4, 2, 1}, {0, 2, 4, 4, -1}};

        float X[nOriSampleBound], Y[nOriSampleBound], angle[nOriSampleBound];
        gls::image<float> PATCH(PATCH_SZ + 1, PATCH_SZ + 1);
        float DX[PATCH_SZ][PATCH_SZ], DY[PATCH_SZ][PATCH_SZ];

        // TODO: should we also add the extended (dsize = 128) case?
        const int dsize = 64;

        float maxSize = 0;
        for (int k = k1; k < k2; k++) {
            maxSize = std::max(maxSize, (*keypoints)[k].size);
        }

        for (int k = k1; k < k2; k++) {
            std::array<SurfHF, NX> dx_t;
            std::array<SurfHF, NY> dy_t;
            KeyPoint& kp = (*keypoints)[k];
            float size = kp.size;
            Point2f center = kp.pt;
            /* The sampling intervals and wavelet sized for selecting an orientation
             and building the keypoint descriptor are defined relative to 's' */
            float s = size * 1.2f / 9.0f;
            /* To find the dominant orientation, the gradients in x and y are
             sampled in a circle of radius 6s using wavelets of size 4s.
             We ensure the gradient wavelet size is even to ensure the
             wavelet pattern is balanced and symmetric around its center */
            int grad_wav_size = 2 * (int) lrint(2 * s);
            if (sum.height < grad_wav_size || sum.width < grad_wav_size) {
                /* when grad_wav_size is too big,
                 * the sampling of gradient will be meaningless
                 * mark keypoint for deletion. */
                kp.size = -1;
                continue;
            }

            float descriptor_dir = 360.f - 90.f;
            resizeHaarPattern(dx_s, &dx_t, 4, grad_wav_size, sum.width);
            resizeHaarPattern(dy_s, &dy_t, 4, grad_wav_size, sum.width);
            int nangle = 0;
            for (int kk = 0; kk < nOriSamples; kk++) {
                // TODO: if we use round instead of lrint the result is slightly different

                int x = (int) lrint(center.x + apt[kk].x * s - (float)(grad_wav_size - 1) / 2);
                int y = (int) lrint(center.y + apt[kk].y * s - (float)(grad_wav_size - 1) / 2);
                if (y < 0 || y >= sum.height - grad_wav_size ||
                    x < 0 || x >= sum.width - grad_wav_size)
                    continue;
                float vx = calcHaarPattern(sum, {x, y}, dx_t);
                float vy = calcHaarPattern(sum, {x, y}, dy_t);
                X[nangle] = vx * aptw[kk];
                Y[nangle] = vy * aptw[kk];
                nangle++;
            }
            if (nangle == 0) {
                // No gradient could be sampled because the keypoint is too
                // near too one or more of the sides of the image. As we
                // therefore cannot find a dominant direction, we skip this
                // keypoint and mark it for later deletion from the sequence.
                kp.size = -1;
                continue;
            }

            // phase( Mat(1, nangle, CV_32F, X), Mat(1, nangle, CV_32F, Y), Mat(1, nangle, CV_32F, angle), true );
            for (int i = 0; i < nangle; i++) {
                float temp = atan2(Y[i], X[i]) * (180 / M_PI);
                if (temp < 0)
                    angle[i] = temp + 360;
                else
                    angle[i] = temp;
            }

            float bestx = 0, besty = 0, descriptor_mod = 0;
            for (float i = 0; i < 360; i += SURF_ORI_SEARCH_INC) {
                float sumx = 0, sumy = 0, temp_mod;
                for (int j = 0; j < nangle; j++) {
                    float d = std::abs(lrint(angle[j]) - i);
                    if (d < ORI_WIN / 2 || d > 360 - ORI_WIN / 2) {
                        sumx += X[j];
                        sumy += Y[j];
                    }
                }
                temp_mod = sumx * sumx + sumy * sumy;
                if (temp_mod > descriptor_mod) {
                    descriptor_mod = temp_mod;
                    bestx = sumx;
                    besty = sumy;
                }
            }
            descriptor_dir = atan2(-besty, bestx);

            kp.angle = descriptor_dir;

            if (!descriptors)
                continue;

            /* Extract a window of pixels around the keypoint of size 20s */
            int win_size = (int)((PATCH_SZ + 1) * s);
            gls::image<float> mwin(win_size, win_size);

            // !upright
            descriptor_dir *= (float)(M_PI / 180);
            float sin_dir = -std::sin(descriptor_dir);
            float cos_dir =  std::cos(descriptor_dir);

            float win_offset = -(float)(win_size - 1) / 2;
            float start_x = center.x + win_offset * cos_dir + win_offset * sin_dir;
            float start_y = center.y - win_offset * sin_dir + win_offset * cos_dir;

            int ncols1 = img.width - 1, nrows1 = img.height - 1;
            for (int i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir) {
                float pixel_x = start_x;
                float pixel_y = start_y;
                for (int j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir) {
                    int ix = std::floor(pixel_x);
                    int iy = std::floor(pixel_y);

                    if ((unsigned)ix < (unsigned)ncols1 &&
                        (unsigned)iy < (unsigned)nrows1) {
                        float a = pixel_x - ix;
                        float b = pixel_y - iy;

                        mwin[i][j] = std::round((img[iy][ix] * (1 - a) + img[iy][ix + 1] * a) * (1 - b) +
                                                (img[iy + 1][ix] * (1 - a) + img[iy + 1][ix + 1] * a) * b);
                    } else {
                        int x = std::clamp((int)std::round(pixel_x), 0, ncols1);
                        int y = std::clamp((int)std::round(pixel_y), 0, nrows1);
                        mwin[i][j] = img[y][x];
                    }
                }
            }

            // Scale the window to size PATCH_SZ so each pixel's size is s. This
            // makes calculating the gradients with wavelets of size 2s easy
            resizeVV(mwin, &PATCH, 0);

            // Calculate gradients in x and y with wavelets of size 2s
            for (int i = 0; i < PATCH_SZ; i++)
                for (int j = 0; j < PATCH_SZ; j++) {
                    float dw = DW[i * PATCH_SZ + j];
                    float vx = (PATCH[i][j + 1] - PATCH[i][j] + PATCH[i + 1][j + 1] - PATCH[i + 1][j]) * dw;
                    float vy = (PATCH[i + 1][j] - PATCH[i][j] + PATCH[i + 1][j + 1] - PATCH[i][j + 1]) * dw;
                    DX[i][j] = vx;
                    DY[i][j] = vy;
                }

            // Construct the descriptor
            for (int kk = 0; kk < dsize; kk++) {
                (*descriptors)[k][kk] = 0;
            }
            float square_mag = 0;

            // 64-bin descriptor
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    int index = 16 * i + 4 * j;

                    for (int y = i * 5; y < i * 5 + 5; y++) {
                        for (int x = j * 5; x < j * 5 + 5; x++) {
                            float tx = DX[y][x], ty = DY[y][x];
                            (*descriptors)[k][index + 0] += tx;
                            (*descriptors)[k][index + 1] += ty;
                            (*descriptors)[k][index + 2] += (float)fabs(tx);
                            (*descriptors)[k][index + 3] += (float)fabs(ty);
                        }
                    }
                    for (int kk = 0; kk < 4; kk++) {
                        float v = (*descriptors)[k][index + kk];
                        square_mag += v * v;
                    }
                }
            }

            // unit vector is essential for contrast invariance
            float scale = (float)(1. / (sqrt(square_mag) + FLT_EPSILON));
            for (int kk = 0; kk < dsize; kk++) {
                (*descriptors)[k][kk] *= scale;
            }
        }
    }

    void run() {
        const int threads = 8;
        ThreadPool threadPool(threads);

        const int K = (int) keypoints->size();
        const int ranges = (int) std::ceil((float) K / threads);

        std::cout << "enqueueing " << threads << " computeRange" << std::endl;
        for (int rr = 0; rr < ranges; rr++) {
            int k1 = ranges * rr;
            int k2 = std::min(ranges * (rr + 1), K);

            threadPool.enqueue([this, k1, k2](){
                computeRange(k1, k2);
            });
        }
    }
};

void SURF_descriptor(const gls::image<float>& srcImg, const gls::image<float>& integralSum, std::vector<KeyPoint>* keypoints, gls::image<float>* descriptors) {
    int N = (int) keypoints->size();
    if (N > 0) {
        SURFInvoker(srcImg, integralSum, keypoints, descriptors).run();
    }
}

float calcDistance(float* p1, float* p2, int n) {
    float distance = 0;
    for (int i = 0; i < n; i++) {
        distance += fabs(p1[i] - p2[i]);
    }
    return distance;
}

void integral(const gls::image<float>& img, gls::image<float>* sum) {
    // Zero the first row and the first column of the sum
    for (int i = 0; i < sum->width; i++) {
        (*sum)[0][i] = 0;
    }
    for (int j = 1; j < sum->height; j++) {
        (*sum)[j][0] = 0;
    }

    for (int j = 1; j < sum->height; j++) {
        for (int i = 1; i < sum->width; i++) {
            (*sum)[j][i] = img[j - 1][i - 1] / SURF_INTEGRAL_BIAS + (*sum)[j][i - 1] + (*sum)[j - 1][i] - (*sum)[j - 1][i - 1];
        }
    }
}

void integral(const gls::image<float>& img, gls::image<double>* sum) {
    // Zero the first row and the first column of the sum
    for (int i = 0; i < sum->width; i++) {
        (*sum)[0][i] = 0;
    }
    for (int j = 1; j < sum->height; j++) {
        (*sum)[j][0] = 0;
    }

    for (int j = 1; j < sum->height; j++) {
        for (int i = 1; i < sum->width; i++) {
            (*sum)[j][i] = img[j - 1][i - 1] / SURF_INTEGRAL_BIAS + (*sum)[j][i - 1] + (*sum)[j - 1][i] - (*sum)[j - 1][i - 1];
        }
    }
}

//void SURF_detectAndCompute(SurfClContext *ctx, const gls::image<float>& img, std::vector<KeyPoint>* keypoints, gls::image<float>::unique_ptr* _descriptors) {
//    bool doDescriptors = _descriptors != nullptr;
//
//    gls::image<float> sum(img.width + 1, img.height + 1);
//    integral(img, &sum);
//
//    int nOctaves = 4;
//    int nOctaveLayers = 2;
//    float hessianThreshold = 800;  // hessian threshold 1300
//
//    fastHessianDetector(ctx, sum, keypoints, nOctaves, nOctaveLayers, hessianThreshold);
//
//    int N = (int)keypoints->size();
//
//    auto descriptors = doDescriptors ? std::make_unique<gls::image<float>>(64, N) : nullptr;
//
//    // we call SURFInvoker in any case, even if we do not need descriptors,
//    // since it computes orientation of each feature.
//    SURFInvoker(img, sum, keypoints, doDescriptors ? descriptors.get() : nullptr).run();
//
//    // remove keypoints that were marked for deletion
//    int j = 0;
//    for (int i = 0; i < N; i++) {
//        if ((*keypoints)[i].size > 0) {
//            if (i > j) {
//                (*keypoints)[j] = (*keypoints)[i];
//                if (doDescriptors) memcpy((*descriptors)[j], (*descriptors)[i], 64 * sizeof(float));
//            }
//            j++;
//        }
//    }
//
//    if (N > j) {
//        N = j;
//        keypoints->resize(N);
//        if (doDescriptors) {
//            *_descriptors = std::make_unique<gls::image<float>>(64, N);
//
//            for (int i = 0; i < N; i++) {
//                memcpy((**_descriptors)[i], (*descriptors)[i], 64 * sizeof(float));
//            }
//        }
//    }
//}

class DMatch {
   public:
    DMatch() : queryIdx(-1), trainIdx(-1), imgIdx(-1), distance(FLT_MAX) {}
    DMatch(int _queryIdx, int _trainIdx, float _distance)
        : queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1), distance(_distance) {}
    DMatch(int _queryIdx, int _trainIdx, int _imgIdx, float _distance)
        : queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx), distance(_distance) {}

    int queryIdx;  // query descriptor index
    int trainIdx;  // train descriptor index
    int imgIdx;    // train image index

    float distance;

    // less is better
    bool operator<(const DMatch& m) const { return distance < m.distance; }
};

struct refineMatch {
    inline bool operator()(const DMatch& mp1, const DMatch& mp2) const {
        if (mp1.distance < mp2.distance) return true;
        if (mp1.distance > mp2.distance) return false;
        // if (mp1.queryIdx < mp2.queryIdx) return true;
        // if (mp1.queryIdx > mp2.queryIdx) return false;
        /*if (mp1.octave > mp2.octave) return true;
        if (mp1.octave < mp2.octave) return false;
        if (mp1.pt.y < mp2.pt.y) return false;
        if (mp1.pt.y > mp2.pt.y) return true;*/
        return mp1.queryIdx < mp2.queryIdx;
    }
};

bool SURF_Detection(gls::OpenCLContext* cLContext, const gls::image<float>& srcIMAGE1, const gls::image<float>& srcIMAGE2,
                     std::vector<Point2f>* matchpoints1, std::vector<Point2f>* matchpoints2, int matches_num) {
    // (1) Convert the format to IMAGE, and calculate the integral image
    gls::cl_image_buffer_2d<float> integralSum1(cLContext->clContext(), srcIMAGE1.width + 1, srcIMAGE1.height + 1);
    gls::cl_image_buffer_2d<float> integralSum2(cLContext->clContext(), srcIMAGE2.width + 1, srcIMAGE2.height + 1);

    // Calculate the integral image
#if USE_OPENCL_INTEGRAL
    clIntegral(cLContext, srcIMAGE1, &integralSum1);
    clIntegral(cLContext, srcIMAGE2, &integralSum2);
#else
    integral(srcIMAGE1, &integralSum1);
    integral(srcIMAGE2, &integralSum2);
#endif

//    const bool measure_integral_error = false;
//    if (measure_integral_error) {
//        float max_err = 0.0001;
//
//        // Double precision integral image for validation
//        gls::image<double> refIntegralSum1(srcIMAGE1.width + 1, srcIMAGE1.height + 1);
//        integral(srcIMAGE1, &refIntegralSum1);
//        int refIntegralSum1Errors = 0;
//
//        for (int j = 0; j < integralSum1.height; j++) {
//            for (int i = 0; i < integralSum1.width; i++) {
//                if (std::abs(integralSum1[j][i] - refIntegralSum1[j][i]) / integralSum1[j][i] > max_err) {
//                    // std::cout << "integralSum1 diff @ " << j << ":" << i << " - " << integralSum1[j][i] << " != " << refIntegralSum1[j][i] << std::endl;
//                    refIntegralSum1Errors++;
//                }
//            }
//        }
//
//        gls::image<double> refIntegralSum2(srcIMAGE2.width + 1, srcIMAGE2.height + 1);
//        integral(srcIMAGE2, &refIntegralSum2);
//        int refIntegralSum2Errors = 0;
//
//        for (int j = 1; j < integralSum2.height; j++) {
//            for (int i = 1; i < integralSum2.width; i++) {
//                if (std::abs(integralSum2[j][i] - refIntegralSum2[j][i]) / integralSum2[j][i] > max_err) {
//                    // std::cout << "integralSum2 diff @ " << j << ":" << i << " - " << integralSum2[j][i] << " != " << refIntegralSum2[j][i] << std::endl;
//                    refIntegralSum2Errors++;
//                }
//            }
//        }
//
//        std::cout << "refIntegralSum1Errors: " << refIntegralSum1Errors << ", refIntegralSum2Errors: " << refIntegralSum2Errors << std::endl;
//    }

    // (2) Detect SURF feature points
    std::vector<KeyPoint> keypoints1, keypoints2;
    float hessianThreshold = 800;  // hessian threshold 1300

    SurfClContext ctx(cLContext);

    // SURF detects feature points
    SURF_detect(&ctx, srcIMAGE1, integralSum1, &keypoints1, hessianThreshold);

    ctx.sumMemory.clear();

    SURF_detect(&ctx, srcIMAGE2, integralSum2, &keypoints2, hessianThreshold);
    printf(" ---------- \n Detected feature points: %ld \t %ld \n", keypoints1.size(),
           keypoints2.size());

//    if (keypoints1.size() >= 10 && keypoints2.size() >= 10) {
//        for (int i = 0; i < 10; i++) {
//            const auto& kp = keypoints1[i];
//            std::cout << "kp1[" << i << "]: " << kp.pt.x << ", " << kp.pt.y
//                      << ", size: " << kp.size
//                      << ", angle: " << kp.angle
//                      << ", response: " << kp.response
//                      << ", octave: " << kp.octave
//                      << ", class_id: " << kp.class_id << std::endl;
//        }
//
//        for (int i = 0; i < 10; i++) {
//            const auto& kp = keypoints2[i];
//            std::cout << "kp2[" << i << "]: " << kp.pt.x << ", " << kp.pt.y
//                      << ", size: " << kp.size
//                      << ", angle: " << kp.angle
//                      << ", response: " << kp.response
//                      << ", octave: " << kp.octave
//                      << ", class_id: " << kp.class_id << std::endl;
//        }
//    }

    // SURF calculates the angle and feature point description matrix
    if (keypoints1.empty() || keypoints2.empty()) {  // If the number of detected feature points is 0, return
        return false;
    }

    /* ********* By limiting the number of feature points, the matching time is shortened and the efficiency is improved *********** */

    int Nk = 2;  // keep the matching point coefficient

    if (true) {
        if (keypoints1.size() > Nk * matches_num) {
            keypoints1.erase(keypoints1.begin() + Nk * matches_num, keypoints1.end());
            printf("keypoints1 erase: %d ", Nk * matches_num);
        }
        if (keypoints2.size() > Nk * matches_num) {
            keypoints2.erase(keypoints2.begin() + Nk * matches_num, keypoints2.end());
            printf("keypoints2 erase: %d\n", Nk * matches_num);
        }
    }
    if (/* DISABLES CODE */ (false)) {
        if (keypoints1.size() > Nk * matches_num) {
            float temp = keypoints1[Nk * matches_num].response;
            int k2index = Nk * matches_num;
            for (int i = 0; i < keypoints2.size(); i++) {
                if (keypoints2[i].response < temp) {
                    k2index = i;
                    break;
                }
            }
            keypoints1.erase(keypoints1.begin() + Nk * matches_num, keypoints1.end());
            printf("keypoints1 erase: %d ", Nk * matches_num);
            keypoints2.erase(keypoints2.begin() + k2index, keypoints2.end());
            printf("keypoints2 erase: %d\n", Nk * matches_num);
        }
    }

    // (3) Feature point description

    gls::image<float> descriptor1(64, (int)keypoints1.size());
    gls::image<float> descriptor2(64, (int)keypoints2.size());
    const auto integralSum1Cpu = integralSum1.mapImage();
    SURF_descriptor(srcIMAGE1, integralSum1Cpu, &keypoints1, &descriptor1);
    const auto integralSum2Cpu = integralSum2.mapImage();
    SURF_descriptor(srcIMAGE2, integralSum2Cpu, &keypoints2, &descriptor2);
    integralSum1.unmapImage(integralSum1Cpu);
    integralSum2.unmapImage(integralSum2Cpu);

    // (4) Match feature points
    std::vector<DMatch> matchedPoints;
    for (int i = 0; i < keypoints1.size(); i++) {
        // start of i line of descriptor1
        float* p1 = descriptor1[i];
        float distance_min = 100;
        int j_min = 0, i_min = 0;
        int j;
        float dx = 0, dy = 0;
        for (j = 0; j < keypoints2.size(); j++) {
            dx = fabs(keypoints1[i].pt.x - keypoints2[j].pt.x);
            dy = fabs(keypoints1[i].pt.y - keypoints2[j].pt.y);

            float* p2 = descriptor2[j];
            // calculate distance
            float distance_t = calcDistance(p1, p2, 64);
            if (distance_t < distance_min) {
                distance_min = distance_t;
                i_min = i;
                j_min = j;
            }
        }
        matchedPoints.push_back(DMatch(i_min, j_min, distance_min));
    }

    std::sort(matchedPoints.begin(), matchedPoints.end(), refineMatch());  // feature point sorting

    // Convert to Point2D format
    int goodPoints = matches_num < (int)matchedPoints.size() ? matches_num : (int)matchedPoints.size();
    for (int i = 0; i < goodPoints; i++) {
        matchpoints1->push_back(Point2f(keypoints1[matchedPoints[i].queryIdx].pt.x,
                                        keypoints1[matchedPoints[i].queryIdx].pt.y));
        matchpoints2->push_back(Point2f(keypoints2[matchedPoints[i].trainIdx].pt.x,
                                        keypoints2[matchedPoints[i].trainIdx].pt.y));
    }

    /* ******* Registration Performance Evaluation - Additional Contents ********* */

    if (matchpoints1->size() && matchpoints2->size())
        return true;
    else
        return false;
}

// RANSAC

bool MatrixMultiplyV(const std::vector<std::vector<float>>& X1, const std::vector<std::vector<float>>& X2,
                     std::vector<std::vector<float>>* Y) {
    int row1 = (int) X1.size();
    int col1 = (int) X1[0].size();
    int row2 = (int) X2.size();
    int col2 = (int) X2[0].size();
    if (col1 != row2) return false;
    Y->resize(row1);
    for (int i = 0; i < row1; i++) (*Y)[i].resize(col2);

    for (int i = 0; i < Y->size(); i++) {
        for (int j = 0; j < (*Y)[0].size(); j++) {
            float sum = 0;
            for (int ki = 0; ki < col1; ki++) {
                sum += X1[i][ki] * X2[ki][j];
            }
            (*Y)[i][j] = sum;
        }
    }
    return true;
}

template <typename TIN, typename TA, typename TB>
void getPerspectiveTransformAB(const TIN& src, const TIN& dst, TA& a, TB& b) {
    int count = (int) src.size();

    for (int i = 0; i < count; i++) {
        a[i][0] = a[i + count][3] = src[i].x;
        a[i][1] = a[i + count][4] = src[i].y;
        a[i][2] = a[i + count][5] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i + count][0] = a[i + count][1] = a[i + count][2] = 0;
        a[i][6] = -src[i].x * dst[i].x;
        a[i][7] = -src[i].y * dst[i].x;
        a[i + count][6] = -src[i].x * dst[i].y;
        a[i + count][7] = -src[i].y * dst[i].y;
        b[i][0] = dst[i].x;
        b[i + count][0] = dst[i].y;
    }
}

template <size_t NN=8>
gls::Vector<NN> getPerspectiveTransformIata(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    gls::Matrix<NN, NN> a;
    gls::Matrix<NN, 1> b;
    getPerspectiveTransformAB(src, dst, a, b);

    gls::Matrix<NN, NN> at = transpose(a);
    gls::Matrix<NN, NN> ata = at * a;
    gls::Vector<NN> atb = at * b;

    return surf::inverse(ata) * atb;
}

template <size_t NN=8>
std::vector<float> getPerspectiveTransformLSM2(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    int count = (int) src.size();

    std::vector<std::vector<float>> a;
    std::vector<std::vector<float>> b;
    a.resize(2 * count);
    b.resize(2 * count);
    for (int i = 0; i < 2 * count; i++) {
        a[i].resize(NN, 0);
        b[i].resize(1, 0);
    }

    getPerspectiveTransformAB(src, dst, a, b);

    std::vector<std::vector<float>> at(NN, std::vector<float>(a.size()));
    for (int i = 0; i < NN; i++) { // transpose of a
        for (int j = 0; j < 2 * count; j++) {
            at[i][j] = a[j][i];
        }
    }

    gls::Matrix<NN, NN> aa;
    gls::Vector<NN> bb;

    std::vector<std::vector<float>> ata(NN, std::vector<float>(NN, 0));
    std::vector<std::vector<float>> atb(NN, std::vector<float>(1, 0));

    if (MatrixMultiplyV(at, a, &ata) && MatrixMultiplyV(at, b, &atb)) {
        for (int i = 0; i < NN; i++) {
            for (int j = 0; j < NN; j++) {
                aa[i][j] = ata[i][j];
            }
        }
        for (int i = 0; i < NN; i++) {
            bb[i] = atb[i][0];
        }
    }

    gls::Vector<NN> trans = surf::inverse(aa) * bb;

    return std::vector<float>(trans.begin(), trans.end());
}

std::vector<float> getRANSAC2(const std::vector<Point2f>& p1, const std::vector<Point2f>& p2, float threshold, int count) {
    std::vector<float> homoV;
    if (p1.size() == 0) return homoV;
    // Calculate the maximum set of interior points
    int max_iters = 2000;
    int iters = max_iters;
    int innerP, max_innerP = 0;
    std::vector<int> innerPvInd;  // Inner point set index - temporary
    std::vector<int> innerPvInd_i;
    std::vector<Point2f> selectP1, selectP2;
    // generate random table
    int selectIndex[2000][4];
    srand(0 /*(unsigned) time(NULL)*/);  // Use time as seed, each time the random number is different
    int pCount = (int)p1.size();

    for (int i = 0; i < 2000; i++) {
        for (int j = 0; j < 4; j++) {
            int ii = 0;
            int temp = 0;
            selectIndex[i][0] = selectIndex[i][1] = selectIndex[i][2] = selectIndex[i][3] =
                pCount + 1;
            while (ii < 4) {
                temp = rand() % pCount;
                if (temp != selectIndex[i][0] && temp != selectIndex[i][1] &&
                    temp != selectIndex[i][2] && temp != selectIndex[i][3]) {
                    selectIndex[i][ii] = temp;
                    ii++;
                }
            }
        }
    }

    int k = 0;
    for (; k < iters; k++) {
        selectP1.push_back(p1[selectIndex[k][0]]);
        selectP1.push_back(p1[selectIndex[k][1]]);
        selectP1.push_back(p1[selectIndex[k][2]]);
        selectP1.push_back(p1[selectIndex[k][3]]);
        selectP2.push_back(p2[selectIndex[k][0]]);
        selectP2.push_back(p2[selectIndex[k][1]]);
        selectP2.push_back(p2[selectIndex[k][2]]);
        selectP2.push_back(p2[selectIndex[k][3]]);

        // Calculate the perspective transformation matrix
        // (currently the singular matrix inverse cannot be solved, and the
        // singular decomposition inversion method will be improved in the future)
        try {
            gls::Vector<8> trans = getPerspectiveTransformIata(selectP1, selectP2);

            // Calculate the model parameter error, if the error is greater than the threshold, discard
            // this set of model parameters
            innerP = 0;
            float u, v, w;
            float errX, errY;
            for (int i = 0; i < p1.size(); i++) {
                errX = errY = 0;
                u = p1[i].x * trans[0] + p1[i].y * trans[1] + trans[2];
                v = p1[i].x * trans[3] + p1[i].y * trans[4] + trans[5];
                w = p1[i].x * trans[6] + p1[i].y * trans[7] + 1;
                errX = fabs(u / w - p2[i].x);
                errY = fabs(v / w - p2[i].y);
                if (threshold > (errX * errX + errY * errY)) {
                    innerP++;
                    innerPvInd.push_back(i);
                }
            }
            if (innerP > max_innerP) {
                max_innerP = innerP;
                innerPvInd_i = innerPvInd;
                // update the number of iterations
                float p = 0.995;
                float ep = (float)(p1.size() - innerP) / p1.size();
                // avoid inf's & nan's
                float num_ = std::max(1.f - p, FLT_MIN);
                float denom_ = 1. - pow(1.f - ep, 4);

                if (denom_ < FLT_MIN)
                    iters = 0;
                else {
                    float num = log(num_);
                    float denom = log(denom_);
                    iters =
                        (denom >= 0 || -num >= max_iters * (-denom) ? max_iters : (int)(num / denom));
                }
            }
            innerPvInd.clear();
        } catch (const std::logic_error& e) {
            printf("Perspective transformation matrix transformation error");
        }

        selectP1.clear();
        selectP2.clear();
    }
    printf(" RANSAC interior point ratio - number of loops: %d %ld %d \t\n ", max_innerP, p1.size(), k);

    // Calculate projection matrix parameters based on interior points

    std::vector<Point2f> _p1, _p2;
    for (int i = 0; i < max_innerP; i++) {
        _p1.push_back(Point2f(p1[innerPvInd_i[i]].x, p1[innerPvInd_i[i]].y));
        _p2.push_back(Point2f(p2[innerPvInd_i[i]].x, p2[innerPvInd_i[i]].y));
    }

    homoV = getPerspectiveTransformLSM2(_p1, _p2);
    return homoV;
}

} // namespace surf

