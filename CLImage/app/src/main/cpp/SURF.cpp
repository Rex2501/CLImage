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

#include "IntegralImage.hpp"

#define USE_OPENCL true
#define USE_OPENCL_INTEGRAL true

template<>
struct std::hash<gls::size> {
    std::size_t operator()(gls::size const& r) const noexcept {
        return r.width ^ r.height;
    }
};

namespace gls {

static const int   SURF_ORI_SEARCH_INC = 5;
static const float SURF_ORI_SIGMA      = 2.5f;
static const float SURF_DESC_SIGMA     = 3.3f;

// Wavelet size at first layer of first octave.
static const int SURF_HAAR_SIZE0 = 9;

// Wavelet size increment between layers. This should be an even number,
// such that the wavelet sizes in an octave are either all even or all odd.
// This ensures that when looking for the neighbours of a sample, the layers
// above and below are aligned correctly.
static const int SURF_HAAR_SIZE_INC = 6;

// Bias value used to optimize the dynamic range of the integral image
static const float SURF_INTEGRAL_BIAS  = 255;

struct SurfHF {
    gls::point p0, p1, p2, p3;
    float w;

    SurfHF() : p0({0, 0}), p1({0, 0}), p2({0, 0}), p3({0, 0}), w(0) {}
};

template <size_t N>
static inline float calcHaarPattern(const gls::image<float>& sum, const gls::point& p, const std::array<SurfHF, N>& f) {
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
static void resizeHaarPattern(const int src[][5], std::array<SurfHF, N>* dst, int oldSize, int newSize) {
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

static void calcDetAndTrace(const gls::image<float>& sum,
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

struct clSurfHF {
    cl_int8 p_dx[3];
    cl_float4 w_dx;

    cl_int8 p_dy[3];
    cl_float4 w_dy;

    cl_int8 p_dxy[4];
    cl_float4 w_dxy;

    clSurfHF(const std::array<SurfHF, 3>& Dx, const std::array<SurfHF, 3>& Dy, const std::array<SurfHF, 4>& Dxy) {
        for (int i = 0; i < 3; i++) {
            p_dx[i] = { Dx[i].p0.x, Dx[i].p0.y, Dx[i].p1.x, Dx[i].p1.y, Dx[i].p2.x, Dx[i].p2.y, Dx[i].p3.x, Dx[i].p3.y };
        }
        w_dx = { Dx[0].w, Dx[1].w, Dx[2].w, 0 };

        for (int i = 0; i < 3; i++) {
            p_dy[i] = { Dy[i].p0.x, Dy[i].p0.y, Dy[i].p1.x, Dy[i].p1.y, Dy[i].p2.x, Dy[i].p2.y, Dy[i].p3.x, Dy[i].p3.y };
        }
        w_dy = { Dy[0].w, Dy[1].w, Dy[2].w, 0 };

        for (int i = 0; i < 4; i++) {
            p_dxy[i] = { Dxy[i].p0.x, Dxy[i].p0.y, Dxy[i].p1.x, Dxy[i].p1.y, Dxy[i].p2.x, Dxy[i].p2.y, Dxy[i].p3.x, Dxy[i].p3.y };
        }
        w_dxy = { Dxy[0].w, Dxy[1].w, Dxy[2].w, Dxy[3].w };
    }
};

struct DetAndTraceHaarPattern {
    static const int NX = 3, NY = 3, NXY = 4;

    std::array<SurfHF, NX> Dx;
    std::array<SurfHF, NY> Dy;
    std::array<SurfHF, NXY> Dxy;

    const int height, width;
    const gls::rectangle margin_crop;

    DetAndTraceHaarPattern(int sum_width, int sum_height, int size, int sampleStep) :
        height(1 + (sum_height - 1 - size) / sampleStep),   // The integral image 'sum' is one pixel bigger than the source image
        width(1 + (sum_width - 1 - size) / sampleStep),
        margin_crop((size / 2) / sampleStep,                // Ignore pixels where some of the kernel is outside the image
                    (size / 2) / sampleStep,
                    width, height) {
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

        if (size > (sum_height - 1) || size > (sum_width - 1))
            return;

        resizeHaarPattern(dx_s, &Dx, 9, size);
        resizeHaarPattern(dy_s, &Dy, 9, size);
        resizeHaarPattern(dxy_s, &Dxy, 9, size);
    }
};

class SURF {
private:
    gls::OpenCLContext* _glsContext;

    std::mutex _keypointsMutex;

    cl::Buffer _surfHFDataBuffer;
    cl::Buffer _keyPointsBuffer;

    std::vector<gls::cl_image_buffer_2d<float>::unique_ptr> _dets;
    std::vector<gls::cl_image_buffer_2d<float>::unique_ptr> _traces;

    const int _nOctaves;
    const int _nOctaveLayers;
    const float _hessianThreshold;

    /* Sampling step along image x and y axes at first octave. This is doubled
       for each additional octave. WARNING: Increasing this improves speed,
       however keypoint extraction becomes unreliable. */
    static const int SAMPLE_STEP0 = 1;

    void calcLayerDetAndTrace(const gls::cl_image_2d<float>& sum, int size, int sampleStep,
                              gls::cl_image_2d<float>* det, gls::cl_image_2d<float>* trace);

    void findMaximaInLayer(const gls::image<float>& sum,
                           const std::array<gls::image<float>*, 3>& dets, const gls::image<float>& trace,
                           const std::array<int, 3>& sizes, std::vector<KeyPoint>* keypoints, int octave,
                           float hessianThreshold, int sampleStep);

    void clCalcLayerDetAndTrace(const gls::cl_image_2d<float>& sum, int size, int sampleStep,
                                gls::cl_image_2d<float>* det, gls::cl_image_2d<float>* trace);

    void clCalcDetAndTrace(const gls::cl_image_2d<float>& sumImage,
                           gls::cl_image_2d<float>* detImage,
                           gls::cl_image_2d<float>* traceImage,
                           const int sampleStep,
                           const DetAndTraceHaarPattern& haarPattern);

    void clCalcDetAndTrace(const gls::cl_image_2d<float>& sumImage,
                           const std::array<gls::cl_image_2d<float>*, 4>& detImage,
                           const std::array<gls::cl_image_2d<float>*, 4>& traceImage,
                           const int sampleStep,
                           const std::array<DetAndTraceHaarPattern, 4>& haarPattern);

    void clFindMaximaInLayer(const gls::cl_image_2d<float>& sumImage,
                             const std::array<const gls::cl_image_2d<float>*, 3>& dets,
                             const gls::cl_image_2d<float>& traceImage,
                             const std::array<int, 3>& sizes, std::vector<KeyPoint>* keypoints, int octave,
                             float hessianThreshold, int sampleStep);

    void Build(const gls::cl_image_2d<float>& sum,
               std::vector<int>& sizes, std::vector<int>& sampleSteps,
               const std::vector<gls::cl_image_2d<float>::unique_ptr>& dets,
               const std::vector<gls::cl_image_2d<float>::unique_ptr>& traces);

    void Find(const gls::cl_image_2d<float>& sum,
              const std::vector<gls::cl_image_2d<float>::unique_ptr>& dets,
              const std::vector<gls::cl_image_2d<float>::unique_ptr>& traces,
              const std::vector<int>& sizes, const std::vector<int>& sampleSteps,
              const std::vector<int>& middleIndices, std::vector<KeyPoint>* keypoints,
              int nOctaveLayers, float hessianThreshold);

    void fastHessianDetector(const gls::cl_image_2d<float>& sum, std::vector<KeyPoint>* keypoints, int nOctaves,
                             int nOctaveLayers, float hessianThreshold);

public:
    SURF(gls::OpenCLContext* glsContext, int width, int height, int nOctaves = 4, int nOctaveLayers = 2, float hessianThreshold = 800) :
        _glsContext(glsContext),
        _nOctaves(nOctaves),
        _nOctaveLayers(nOctaveLayers),
        _hessianThreshold(hessianThreshold) {
            int nTotalLayers = (nOctaveLayers + 2) * nOctaves;

            if (_dets.size() != nTotalLayers) {
                printf("resizing dets and traces vectors to %d\n", nTotalLayers);
                _dets.resize(nTotalLayers);
                _traces.resize(nTotalLayers);
            }

            // Allocate space for each layer
            int index = 0, step = SAMPLE_STEP0;

            for (int octave = 0; octave < nOctaves; octave++) {
                for (int layer = 0; layer < nOctaveLayers + 2; layer++) {
                    /* The integral image sum is one pixel bigger than the source image*/
                    if (_dets[index] == nullptr) {
                        _dets[index] = std::make_unique<gls::cl_image_buffer_2d<float>>(_glsContext->clContext(), width / step, height / step);
                        _traces[index] = std::make_unique<gls::cl_image_buffer_2d<float>>(_glsContext->clContext(), width / step, height / step);
                    }
                    index++;
                }
                step *= 2;
            }
        }

    void descriptor(const gls::image<float>& srcImg, const gls::image<float>& integralSum, std::vector<KeyPoint>* keypoints, gls::image<float>* descriptors);

    void detect(const gls::cl_image_2d<float>& integralSum, std::vector<KeyPoint>* keypoints) {
        fastHessianDetector(integralSum, keypoints, _nOctaves, _nOctaveLayers, _hessianThreshold);
    }

    void detectAndCompute(const gls::image<float>& img, std::vector<KeyPoint>* keypoints, gls::image<float>::unique_ptr* _descriptors);
};

void SURF::clCalcDetAndTrace(const gls::cl_image_2d<float>& sumImage,
                             gls::cl_image_2d<float>* detImage,
                             gls::cl_image_2d<float>* traceImage,
                             const int sampleStep,
                             const DetAndTraceHaarPattern& haarPattern) {
    // Load the shader source
    const auto program = _glsContext->loadProgram("SURF");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // sumImage
                                    cl::Image2D,  // detImage
                                    cl::Image2D,  // traceImage
                                    cl_int2,      // margin
                                    int,          // sampleStep
                                    cl::Buffer    // surfHFData
                                    >(program, "calcDetAndTrace");

    const clSurfHF surfHFData(haarPattern.Dx, haarPattern.Dy, haarPattern.Dxy);

    if (_surfHFDataBuffer.get() == 0) {
        _surfHFDataBuffer = cl::Buffer(CL_MEM_READ_ONLY, sizeof(clSurfHF));
    }
    cl::enqueueWriteBuffer(_surfHFDataBuffer, false, 0, sizeof(clSurfHF), &surfHFData);

    const auto& margin_crop = haarPattern.margin_crop;

    // Schedule the kernel on the GPU
    kernel(
#if __APPLE__
           gls::OpenCLContext::buildEnqueueArgs(margin_crop.width, margin_crop.height),
#else
           cl::EnqueueArgs(cl::NDRange(margin_crop.width, margin_crop.height), cl::NDRange(32, 32)),
#endif
           sumImage.getImage2D(), detImage->getImage2D(), traceImage->getImage2D(),
           { margin_crop.x, margin_crop.y }, sampleStep, _surfHFDataBuffer);
}

void SURF::clCalcDetAndTrace(const gls::cl_image_2d<float>& sumImage,
                             const std::array<gls::cl_image_2d<float>*, 4>& detImage,
                             const std::array<gls::cl_image_2d<float>*, 4>& traceImage,
                             const int sampleStep,
                             const std::array<DetAndTraceHaarPattern, 4>& haarPattern) {
    // Load the shader source
    const auto program = _glsContext->loadProgram("SURF");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // sumImage
                                    cl::Image2D, cl::Image2D, cl::Image2D, cl::Image2D,  // detImage
                                    cl::Image2D, cl::Image2D, cl::Image2D, cl::Image2D,  // traceImage
                                    cl_int4,      // margin
                                    cl_int,      // sampleStep
                                    cl::Buffer    // surfHFData
                                    >(program, "calcDetAndTrace4");

    const std::array<clSurfHF, 4> surfHFData = {
        clSurfHF(haarPattern[0].Dx, haarPattern[0].Dy, haarPattern[0].Dxy),
        clSurfHF(haarPattern[1].Dx, haarPattern[1].Dy, haarPattern[1].Dxy),
        clSurfHF(haarPattern[2].Dx, haarPattern[2].Dy, haarPattern[2].Dxy),
        clSurfHF(haarPattern[3].Dx, haarPattern[3].Dy, haarPattern[3].Dxy)
    };

    if (_surfHFDataBuffer.get() == 0) {
        _surfHFDataBuffer = cl::Buffer(CL_MEM_READ_ONLY, sizeof(surfHFData));
    }
    cl::enqueueWriteBuffer(_surfHFDataBuffer, false, 0, sizeof(surfHFData), &surfHFData);

    kernel(
#if __APPLE__
           gls::OpenCLContext::buildEnqueueArgs(haarPattern[0].margin_crop.width, haarPattern[0].margin_crop.height),
#else
           cl::EnqueueArgs(cl::NDRange(haarPattern[0].margin_crop.width, haarPattern[0].margin_crop.height), cl::NDRange(32, 32)),
#endif
           sumImage.getImage2D(),
           detImage[0]->getImage2D(), detImage[1]->getImage2D(), detImage[2]->getImage2D(), detImage[3]->getImage2D(),
           traceImage[0]->getImage2D(), traceImage[1]->getImage2D(), traceImage[2]->getImage2D(), traceImage[3]->getImage2D(),
           { haarPattern[0].margin_crop.x, haarPattern[1].margin_crop.x, haarPattern[2].margin_crop.x, haarPattern[3].margin_crop.x },
           sampleStep, _surfHFDataBuffer);
}

void SURF::calcLayerDetAndTrace(const gls::cl_image_2d<float>& sum, int size, int sampleStep,
                                gls::cl_image_2d<float>* det, gls::cl_image_2d<float>* trace) {
    DetAndTraceHaarPattern haarPattern(sum.width, sum.height, size, sampleStep);

    gls::image<float> sumCpu = sum.mapImage();
    gls::image<float> detCpuFull = det->mapImage();
    gls::image<float> traceCpuFull = trace->mapImage();

    gls::image<float> detCpu = gls::image<float>(detCpuFull, haarPattern.margin_crop);
    gls::image<float> traceCpu = gls::image<float>(traceCpuFull, haarPattern.margin_crop);

    for (int y = 0; y < haarPattern.height; y++) {
        for (int x = 0; x < haarPattern.width; x++) {
            calcDetAndTrace(sumCpu, &detCpu, &traceCpu, x, y, sampleStep, haarPattern.Dx, haarPattern.Dy, haarPattern.Dxy);
        }
    }

    sum.unmapImage(sumCpu);
    det->unmapImage(detCpuFull);
    trace->unmapImage(traceCpuFull);
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

void SURF::Build(const gls::cl_image_2d<float>& sum,
                 std::vector<int>& sizes, std::vector<int>& sampleSteps,
                 const std::vector<gls::cl_image_2d<float>::unique_ptr>& dets,
                 const std::vector<gls::cl_image_2d<float>::unique_ptr>& traces) {
    int N = (int) sizes.size();
    std::cout << "enqueueing " << N << " calcLayerDetAndTrace" << std::endl;

    ThreadPool threadPool(8);

    const int layers = _nOctaveLayers + 2;

    assert(_nOctaves * layers == N);

    for (int octave = 0; octave < _nOctaves; octave++) {
        const int i = octave * layers;

        std::cout << "calcLayerDetAndTrace " << i << " of " << N << " @ " << dets[i]->width << ", " << dets[i]->width << std::endl;

#if __ANDROID__
        std::array<DetAndTraceHaarPattern, 4> haarPatterns = {
            DetAndTraceHaarPattern(sum.width, sum.height, sizes[i], sampleSteps[i]),
            DetAndTraceHaarPattern(sum.width, sum.height, sizes[i+1], sampleSteps[i+1]),
            DetAndTraceHaarPattern(sum.width, sum.height, sizes[i+2], sampleSteps[i+2]),
            DetAndTraceHaarPattern(sum.width, sum.height, sizes[i+3], sampleSteps[i+3])
        };

        clCalcDetAndTrace(sum, { dets[i].get(), dets[i + 1].get(), dets[i + 2].get(), dets[i + 3].get() },
                          { traces[i].get(), traces[i + 1].get(), traces[i + 2].get(), traces[i + 3].get() },
                          sampleSteps[i], haarPatterns);
#else
        for (int layer = 0; layer < layers; layer++) {
            const int i = octave * layers + layer;
#if USE_OPENCL
            DetAndTraceHaarPattern haarPattern(sum.width, sum.height, sizes[i], sampleSteps[i]);
            clCalcDetAndTrace(sum, dets[i].get(), traces[i].get(), sampleSteps[i], haarPattern);
#else
            threadPool.enqueue([&sum, &dets, &traces, &sizes, &sampleSteps, i, this](){
                calcLayerDetAndTrace(sum, sizes[i], sampleSteps[i], dets[i].get(), traces[i].get());
            });
#endif
        }
#endif
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

void SURF::clFindMaximaInLayer(const gls::cl_image_2d<float>& sumImage,
                               const std::array<const gls::cl_image_2d<float>*, 3>& dets,
                               const gls::cl_image_2d<float>& traceImage,
                               const std::array<int, 3>& sizes,
                               std::vector<KeyPoint>* keypoints,
                               int octave, float hessianThreshold, int sampleStep) {
    // Load the shader source
    const auto program = _glsContext->loadProgram("SURF");

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

    if (_keyPointsBuffer() == 0) {
        _keyPointsBuffer = cl::Buffer(CL_MEM_READ_WRITE, sizeof(KeyPointMaxima));
    }

    // The integral image 'sum' is one pixel bigger than the source image
    const int layer_height = (sumImage.height - 1) / sampleStep;
    const int layer_width = (sumImage.width - 1) / sampleStep;

    // Ignore pixels without a 3x3x3 neighbourhood in the layer above
    const int margin = (sizes[2] / 2) / sampleStep + 1;

    // Schedule the kernel on the GPU
    kernel(
#if __APPLE__
           gls::OpenCLContext::buildEnqueueArgs(layer_width - 2 * margin, layer_height - 2 * margin),
#else
           cl::EnqueueArgs(cl::NDRange(layer_width - 2 * margin, layer_height - 2 * margin), cl::NDRange(32, 32)),
#endif
           dets[0]->getImage2D(), dets[1]->getImage2D(), dets[2]->getImage2D(),
           traceImage.getImage2D(),
           { sizes[0], sizes[1], sizes[2] }, _keyPointsBuffer,
           margin, octave, hessianThreshold, sampleStep);
}

void SURF::findMaximaInLayer(const gls::image<float>& sum,
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
                        std::lock_guard<std::mutex> guard(_keypointsMutex);
                        keypoints->push_back(kpt);
                        keyPointMaxima++;
                    }
                }
            }
        }
    }
    std::cout << "keyPointMaxima: " << keyPointMaxima << std::endl;
}

void SURF::Find(const gls::cl_image_2d<float>& sum,
                const std::vector<gls::cl_image_2d<float>::unique_ptr>& dets,
                const std::vector<gls::cl_image_2d<float>::unique_ptr>& traces,
                const std::vector<int>& sizes, const std::vector<int>& sampleSteps,
                const std::vector<int>& middleIndices, std::vector<KeyPoint>* keypoints,
                int nOctaveLayers, float hessianThreshold) {
    ThreadPool threadPool(8);

    int M = (int) middleIndices.size();
    std::cout << "enqueueing " << M << " findMaximaInLayer" << std::endl;
    for (int i = 0; i < M; i++) {
        const int layer = middleIndices[i];
        const int octave = i / nOctaveLayers;

        std::cout << "findiong maxima in layers " << layer-1 << ", " << layer << ", " << layer+1 << std::endl;

#if USE_OPENCL
        const std::array<const gls::cl_image_2d<float>*, 3> detImages = {
            dets[layer-1].get(),
            dets[layer].get(),
            dets[layer+1].get()
        };

        const auto traceImage = traces[layer].get();

        clFindMaximaInLayer(sum, detImages, *traceImage,
                            { sizes[layer-1], sizes[layer], sizes[layer+1] },
                            keypoints, octave, hessianThreshold, sampleSteps[layer]);
#else
        threadPool.enqueue([&sum, &dets, &traces, &sizes, &sampleSteps, &keypoints, layer, octave, hessianThreshold, this](){
            const auto sumImage = sum.mapImage();
            auto dets0 = dets[layer-1]->mapImage();
            auto dets1 = dets[layer]->mapImage();
            auto dets2 = dets[layer+1]->mapImage();

            auto traceImage = traces[layer]->mapImage();

            findMaximaInLayer(sumImage, { &dets0, &dets1, &dets2 },
                              traceImage, { sizes[layer-1], sizes[layer], sizes[layer+1] },
                              keypoints, octave, hessianThreshold, sampleSteps[layer]);

            sum.unmapImage(sumImage);
            dets[layer-1]->unmapImage(dets0);
            dets[layer]->unmapImage(dets1);
            dets[layer+1]->unmapImage(dets2);
            traces[layer]->unmapImage(traceImage);
        });
#endif
    }

#if USE_OPENCL
    // Collect results
    const auto keyPointMaxima = (KeyPointMaxima *) cl::enqueueMapBuffer(_keyPointsBuffer, true, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(KeyPointMaxima));

    std::cout << "keyPointMaxima: " << keyPointMaxima->count << std::endl;
    std::span<KeyPoint> newElements(keyPointMaxima->keyPoints, std::min(keyPointMaxima->count, KeyPointMaxima::MaxCount));
    keypoints->insert(end(*keypoints), begin(newElements), end(newElements));

    // Reset count
    keyPointMaxima->count = 0;

    cl::enqueueUnmapMemObject(_keyPointsBuffer, (void*)keyPointMaxima);
#endif
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

void SURF::fastHessianDetector(const gls::cl_image_2d<float>& sum, std::vector<KeyPoint>* keypoints, int nOctaves, int nOctaveLayers, float hessianThreshold) {
    int nTotalLayers = (nOctaveLayers + 2) * nOctaves;
    int nMiddleLayers = nOctaveLayers * nOctaves;

    std::vector<int> sizes(nTotalLayers);
    std::vector<int> sampleSteps(nTotalLayers);
    std::vector<int> middleIndices(nMiddleLayers);

    keypoints->clear();

    // Calculate properties of each layer
    int index = 0, middleIndex = 0, step = SAMPLE_STEP0;

    for (int octave = 0; octave < nOctaves; octave++) {
        for (int layer = 0; layer < nOctaveLayers + 2; layer++) {
            sizes[index] = (SURF_HAAR_SIZE0 + SURF_HAAR_SIZE_INC * layer) << octave;
            sampleSteps[index] = step;

            if (0 < layer && layer <= nOctaveLayers) {
                middleIndices[middleIndex++] = index;
            }
            index++;
        }
        step *= 2;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Calculate hessian determinant and trace samples in each layer
    Build(sum, sizes, sampleSteps, _dets, _traces);

    // Find maxima in the determinant of the hessian
    Find(sum, _dets, _traces, sizes, sampleSteps, middleIndices, keypoints, nOctaveLayers, hessianThreshold);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    printf("Features Finding Time: %.2fms\n", elapsed_time_ms);

    sort(keypoints->begin(), keypoints->end(), KeypointGreater());
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
        const int dx_s[NX][5] = {
            {0, 0, 2, 4, -1},
            {2, 0, 4, 4, 1}
        };
        const int dy_s[NY][5] = {
            {0, 0, 4, 2, 1},
            {0, 2, 4, 4, -1}
        };

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
            resizeHaarPattern(dx_s, &dx_t, 4, grad_wav_size);
            resizeHaarPattern(dy_s, &dy_t, 4, grad_wav_size);
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
            float scale = (float)(1. / (std::sqrt(square_mag) + FLT_EPSILON));
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

void SURF::descriptor(const gls::image<float>& srcImg, const gls::image<float>& integralSum, std::vector<KeyPoint>* keypoints, gls::image<float>* descriptors) {
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

void SURF::detectAndCompute(const gls::image<float>& img, std::vector<KeyPoint>* keypoints, gls::image<float>::unique_ptr* _descriptors) {
    bool doDescriptors = _descriptors != nullptr;

    gls::cl_image_buffer_2d<float> sum(_glsContext->clContext(), img.width + 1, img.height + 1);
    gls::clIntegral(_glsContext, img, &sum, SURF_INTEGRAL_BIAS);

    fastHessianDetector(sum, keypoints, _nOctaves, _nOctaveLayers, _hessianThreshold);

    int N = (int)keypoints->size();

    auto descriptors = doDescriptors ? std::make_unique<gls::image<float>>(64, N) : nullptr;

    // we call SURFInvoker in any case, even if we do not need descriptors,
    // since it computes orientation of each feature.
    printf("mapping intregral images...\n");

    const auto integralSumCpu = sum.mapImage();

    descriptor(img, integralSumCpu, keypoints, _descriptors->get());

    printf("unmapping intregral images...\n");

    // remove keypoints that were marked for deletion
    int j = 0;
    for (int i = 0; i < N; i++) {
        if ((*keypoints)[i].size > 0) {
            if (i > j) {
                (*keypoints)[j] = (*keypoints)[i];
                if (doDescriptors) memcpy((*descriptors)[j], (*descriptors)[i], 64 * sizeof(float));
            }
            j++;
        }
    }

    if (N > j) {
        N = j;
        keypoints->resize(N);
        if (doDescriptors) {
            *_descriptors = std::make_unique<gls::image<float>>(64, N);

            for (int i = 0; i < N; i++) {
                memcpy((**_descriptors)[i], (*descriptors)[i], 64 * sizeof(float));
            }
        }
    }
}

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

//bool MatchImages(gls::OpenCLContext* cLContext, const gls::image<float>& inputImage1, const gls::image<float>& inputImage2,
//                 std::vector<Point2f>* matchpoints1, std::vector<Point2f>* matchpoints2, int matches_num) {
//
//    return true;
//}

double timeDiff(std::chrono::steady_clock::time_point t_start,
                std::chrono::steady_clock::time_point t_end) {
    return std::chrono::duration<double, std::milli>(t_end-t_start).count();
}

bool SURF_Detection(gls::OpenCLContext* cLContext, const gls::image<float>& srcIMAGE1, const gls::image<float>& srcIMAGE2,
                    std::vector<Point2f>* matchpoints1, std::vector<Point2f>* matchpoints2, int matches_num) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // (1) Convert the format to IMAGE, and calculate the integral image
    // Calculate the integral image
    gls::cl_image_buffer_2d<float> integralSum1(cLContext->clContext(), srcIMAGE1.width + 1, srcIMAGE1.height + 1);
    gls::cl_image_buffer_2d<float> integralSum2(cLContext->clContext(), srcIMAGE2.width + 1, srcIMAGE2.height + 1);

#if USE_OPENCL_INTEGRAL
    clIntegral(cLContext, srcIMAGE1, &integralSum1, SURF_INTEGRAL_BIAS);
    clIntegral(cLContext, srcIMAGE2, &integralSum2, SURF_INTEGRAL_BIAS);
#else
    auto integralSum1CPU = integralSum1.mapImage();
    auto integralSum2CPU = integralSum2.mapImage();

    integral(srcIMAGE1, &integralSum1CPU, SURF_INTEGRAL_BIAS);
    integral(srcIMAGE2, &integralSum2CPU, SURF_INTEGRAL_BIAS);

    integralSum1.unmapImage(integralSum1CPU);
    integralSum2.unmapImage(integralSum2CPU);
#endif

    auto t_integral = std::chrono::high_resolution_clock::now();
    printf("--> Integral Image Time: %.2fms\n", timeDiff(t_start, t_integral));

    // (2) Detect SURF feature points
    std::vector<KeyPoint> keypoints1, keypoints2;
    // float hessianThreshold = 800;  // hessian threshold 1300

    SURF surf(cLContext, srcIMAGE1.width, srcIMAGE1.height, /*nOctaves=*/ 4, /*nOctaveLayers=*/ 2, /*hessianThreshold=*/ 800);

    auto t_surf = std::chrono::high_resolution_clock::now();
    printf("--> SURF Creation Time: %.2fms\n", timeDiff(t_integral, t_surf));

    // SURF detects feature points
    surf.detect(integralSum1, &keypoints1);
    surf.detect(integralSum2, &keypoints2);

    auto t_detect = std::chrono::high_resolution_clock::now();
    printf("--> SURF Detection Time: %.2fms\n", timeDiff(t_surf, t_detect));

    printf(" ---------- \n Detected feature points: %ld \t %ld \n", keypoints1.size(),
           keypoints2.size());

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

    auto t_erase = std::chrono::high_resolution_clock::now();
    printf("--> Keypoint Erase Time: %.2fms\n", timeDiff(t_detect, t_erase));

    // (3) Feature point description

    gls::image<float> descriptor1(64, (int)keypoints1.size());
    gls::image<float> descriptor2(64, (int)keypoints2.size());

    const auto integralSum1Cpu = integralSum1.mapImage();
    const auto integralSum2Cpu = integralSum2.mapImage();

    auto t_mapped = std::chrono::high_resolution_clock::now();
    printf("--> Integral Image Mapping Time: %.2fms\n", timeDiff(t_erase, t_mapped));

    surf.descriptor(srcIMAGE1, integralSum1Cpu, &keypoints1, &descriptor1);
    surf.descriptor(srcIMAGE2, integralSum2Cpu, &keypoints2, &descriptor2);

    auto t_descriptor = std::chrono::high_resolution_clock::now();
    printf("--> SURF Descriptor Time: %.2fms\n", timeDiff(t_erase, t_descriptor));

    integralSum1.unmapImage(integralSum1Cpu);
    integralSum2.unmapImage(integralSum2Cpu);

    auto t_unmapping = std::chrono::high_resolution_clock::now();
    printf("--> Integral Image Unmapping Time: %.2fms\n", timeDiff(t_descriptor, t_unmapping));

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

    auto t_end = std::chrono::high_resolution_clock::now();

    printf("--> Keypoint Matching & Sorting Time: %.2fms\n", timeDiff(t_unmapping, t_end));

    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    printf("--> Features Finding Time: %.2fms\n", elapsed_time_ms);

    /* ******* Registration Performance Evaluation - Additional Contents ********* */

    if (matchpoints1->size() && matchpoints2->size())
        return true;
    else
        return false;
}

void clRegisterAndFuse(gls::OpenCLContext* cLContext,
                       const gls::cl_image_2d<gls::rgba_pixel>& inputImage0,
                       const gls::cl_image_2d<gls::rgba_pixel>& inputImage1,
                       gls::cl_image_2d<gls::rgba_pixel>* outputImage,
                       const gls::Matrix<3, 3>& homography) {
    // Load the shader source
    const auto program = cLContext->loadProgram("SURF");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,        // inputImage0
                                    cl::Image2D,        // inputImage1
                                    cl::Image2D,        // outputImage
                                    gls::Matrix<3, 3>,  // homography
                                    cl::Sampler         // linear_sampler
                                    >(program, "registerAndFuse");

    const auto linear_sampler = cl::Sampler(cLContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

    kernel(
#if __APPLE__
           gls::OpenCLContext::buildEnqueueArgs(inputImage0.width, inputImage0.height),
#else
           cl::EnqueueArgs(cl::NDRange(inputImage0.width, inputImage0.height), cl::NDRange(32, 32)),
#endif
           inputImage0.getImage2D(), inputImage1.getImage2D(), outputImage->getImage2D(), homography, linear_sampler);
}

} // namespace surf
