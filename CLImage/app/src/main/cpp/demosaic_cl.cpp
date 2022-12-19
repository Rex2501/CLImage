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

#include "demosaic.hpp"

#include <iomanip>

#include "gls_cl.hpp"
#include "gls_cl_image.hpp"

/*
 OpenCL RAW Image Demosaic.
 NOTE: This code can throw exceptions, to facilitate debugging no exception handler is provided, so things can crash in place.
 */

void scaleRawData(gls::OpenCLContext* glsContext,
                 const gls::cl_image_2d<gls::luma_pixel_16>& rawImage,
                 gls::cl_image_2d<gls::luma_pixel_float>* scaledRawImage,
                 BayerPattern bayerPattern, gls::Vector<4> scaleMul, float blackLevel) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    cl::Image2D,  // scaledRawImage
                                    int,          // bayerPattern
                                    cl_float4,    // scaleMul
                                    float         // blackLevel
                                    >(program, "scaleRawData");

    // Work on one Quad (2x2) at a time
    kernel(gls::OpenCLContext::buildEnqueueArgs(scaledRawImage->width/2, scaledRawImage->height/2),
           rawImage.getImage2D(), scaledRawImage->getImage2D(), bayerPattern, {scaleMul[0], scaleMul[1], scaleMul[2], scaleMul[3]}, blackLevel);
}

void rawImageGradient(gls::OpenCLContext* glsContext,
                      const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                      gls::Vector<2> rawVariance,
                      gls::cl_image_2d<gls::luma_alpha_pixel_float>* gradientImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    cl_float2,    // rawVariance
                                    cl::Image2D   // gradientImage
                                    >(program, "rawImageGradient");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(gradientImage->width, gradientImage->height),
           rawImage.getImage2D(), { rawVariance[0], rawVariance[1] }, gradientImage->getImage2D());
}

void interpolateGreen(gls::OpenCLContext* glsContext,
                      const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                      const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                      gls::cl_image_2d<gls::luma_pixel_float>* greenImage,
                      BayerPattern bayerPattern, gls::Vector<2> greenVariance) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    cl::Image2D,  // gradientImage
                                    cl::Image2D,  // greenImage
                                    int,          // bayerPattern
                                    cl_float2     // greenVariance
                                    >(program, "interpolateGreen");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(greenImage->width, greenImage->height),
           rawImage.getImage2D(), gradientImage.getImage2D(), greenImage->getImage2D(), bayerPattern, { greenVariance[0], greenVariance[1] });
}

void interpolateRedBlue(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                        const gls::cl_image_2d<gls::luma_pixel_float>& greenImage,
                        const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                        gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                        BayerPattern bayerPattern, gls::Vector<2> redVariance, gls::Vector<2> blueVariance) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    cl::Image2D,  // greenImage
                                    cl::Image2D,  // gradientImage
                                    cl::Image2D,  // rgbImage
                                    int,          // bayerPattern
                                    cl_float2,    // redVariance
                                    cl_float2     // blueVariance
                                    >(program, "interpolateRedBlue");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbImage->width / 2, rgbImage->height / 2),
           rawImage.getImage2D(), greenImage.getImage2D(), gradientImage.getImage2D(),
           rgbImage->getImage2D(), bayerPattern,
           { redVariance[0], redVariance[1] }, { blueVariance[0], blueVariance[1] });
}

void interpolateRedBlueAtGreen(gls::OpenCLContext* glsContext,
                               const gls::cl_image_2d<gls::rgba_pixel_float>& rgbImageIn,
                               const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                               gls::cl_image_2d<gls::rgba_pixel_float>* rgbImageOut,
                               BayerPattern bayerPattern, gls::Vector<2> redVariance, gls::Vector<2> blueVariance) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rgbImageIn
                                    cl::Image2D,  // gradientImage
                                    cl::Image2D,  // rgbImageOut
                                    int,          // bayerPattern
                                    cl_float2,    // redVariance
                                    cl_float2     // blueVariance
                                    >(program, "interpolateRedBlueAtGreen");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbImageOut->width / 2, rgbImageOut->height / 2),
           rgbImageIn.getImage2D(), gradientImage.getImage2D(),
           rgbImageOut->getImage2D(), bayerPattern,
           { redVariance[0], redVariance[1] }, { blueVariance[0], blueVariance[1] });
}

void malvar(gls::OpenCLContext* glsContext,
            const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
            const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
            gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
            BayerPattern bayerPattern, gls::Vector<2> redVariance,
            gls::Vector<2> greenVariance, gls::Vector<2> blueVariance) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    cl::Image2D,  // gradientImage
                                    cl::Image2D,  // rgbImage
                                    int,          // bayerPattern
                                    cl_float2,    // redVariance
                                    cl_float2,    // greenVariance
                                    cl_float2     // blueVariance
                                    >(program, "malvar");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbImage->width, rgbImage->height),
           rawImage.getImage2D(), gradientImage.getImage2D(), rgbImage->getImage2D(),
           bayerPattern,
           { redVariance[0], redVariance[1] },
           { greenVariance[0], greenVariance[1] },
           { blueVariance[0], blueVariance[1] });
}

void fasteDebayer(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                  gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                  BayerPattern bayerPattern) {
    assert(rawImage.width == 2 * rgbImage->width && rawImage.height == 2 * rgbImage->height);

    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    cl::Image2D,  // rgbImage
                                    int           // bayerPattern
                                    >(program, "fastDebayer");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbImage->width, rgbImage->height),
           rawImage.getImage2D(), rgbImage->getImage2D(), bayerPattern);
}

void rawNoiseStatistics(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                        BayerPattern bayerPattern,
                        gls::cl_image_2d<gls::rgba_pixel_float>* meanImage,
                        gls::cl_image_2d<gls::rgba_pixel_float>* varImage) {
    assert(rawImage.width == 2 * meanImage->width && rawImage.height == 2 * meanImage->height);
    assert(varImage->width == meanImage->width && varImage->height == meanImage->height);

    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    int,          // bayerPattern
                                    cl::Image2D,  // meanImage
                                    cl::Image2D   // varImage
                                    >(program, "rawNoiseStatistics");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(meanImage->width, meanImage->height),
           rawImage.getImage2D(), bayerPattern, meanImage->getImage2D(), varImage->getImage2D());
}

template <typename T1, typename T2>
void applyKernel(gls::OpenCLContext* glsContext, const std::string& kernelName,
                 const gls::cl_image_2d<T1>& inputImage,
                 gls::cl_image_2d<T2>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl::Image2D   // outputImage
                                    >(program, kernelName);

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), outputImage->getImage2D());
}

template
void applyKernel(gls::OpenCLContext* glsContext, const std::string& kernelName,
                 const gls::cl_image_2d<gls::luma_pixel_float>& inputImage,
                 gls::cl_image_2d<gls::luma_pixel_float>* outputImage);

template
void applyKernel(gls::OpenCLContext* glsContext, const std::string& kernelName,
                 const gls::cl_image_2d<gls::luma_alpha_pixel_float>& inputImage,
                 gls::cl_image_2d<gls::luma_alpha_pixel_float>* outputImage);

template
void applyKernel(gls::OpenCLContext* glsContext, const std::string& kernelName,
                 const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                 gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

template
void applyKernel(gls::OpenCLContext* glsContext, const std::string& kernelName,
                 const gls::cl_image_2d<gls::luma_pixel_float>& inputImage,
                 gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

template <typename T>
void resampleImage(gls::OpenCLContext* glsContext, const std::string& kernelName, const gls::cl_image_2d<T>& inputImage,
                   gls::cl_image_2d<T>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    const auto linear_sampler = cl::Sampler(glsContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl::Image2D,  // outputImage
                                    cl::Sampler
                                    >(program, kernelName);

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), outputImage->getImage2D(), linear_sampler);
}

template
void resampleImage(gls::OpenCLContext* glsContext, const std::string& kernelName, const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                   gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

template
void resampleImage(gls::OpenCLContext* glsContext, const std::string& kernelName, const gls::cl_image_2d<gls::luma_alpha_pixel_float>& inputImage,
                   gls::cl_image_2d<gls::luma_alpha_pixel_float>* outputImage);

template <typename T>
void subtractNoiseImage(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<T>& inputImage,
                        const gls::cl_image_2d<T>& inputImage1,
                        const gls::cl_image_2d<T>& inputImageDenoised1,
                        const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                        float luma_weight, float sharpening, const gls::Vector<2>& nlf, gls::cl_image_2d<T>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    const auto linear_sampler = cl::Sampler(glsContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl::Image2D,  // inputImage1
                                    cl::Image2D,  // inputImageDenoised1
                                    cl::Image2D,  // gradientImage
                                    float,        // luma_weight
                                    float,        // sharpening
                                    cl_float2,    // nlf
                                    cl::Image2D,  // outputImage
                                    cl::Sampler   // linear_sampler
                                    >(program, "subtractNoiseImage");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), inputImage1.getImage2D(),
           inputImageDenoised1.getImage2D(), gradientImage.getImage2D(),
           luma_weight, sharpening, { nlf[0], nlf[1] }, outputImage->getImage2D(), linear_sampler);
}

template
void subtractNoiseImage(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                        const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage1,
                        const gls::cl_image_2d<gls::rgba_pixel_float>& inputImageDenoised1,
                        const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                        float luma_weight, float sharpening, const gls::Vector<2>& nlf,
                        gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void transformImage(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                    gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                    const gls::Matrix<3, 3>& transform) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    struct Matrix3x3 {
        cl_float3 m[3];
    } clTransform = {{
        { transform[0][0], transform[0][1], transform[0][2] },
        { transform[1][0], transform[1][1], transform[1][2] },
        { transform[2][0], transform[2][1], transform[2][2] }
    }};

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // linearImage
                                    cl::Image2D,  // rgbImage
                                    Matrix3x3     // transform
                                    >(program, "transformImage");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbImage->width, rgbImage->height),
           linearImage.getImage2D(), rgbImage->getImage2D(), clTransform);
}

void convertTosRGB(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                  const gls::cl_image_2d<gls::luma_pixel_float>& ltmMaskImage,
                  gls::cl_image_2d<gls::rgba_pixel>* rgbImage,
                  const DemosaicParameters& demosaicParameters) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    const auto& transform = demosaicParameters.rgb_cam;

    struct Matrix3x3 {
        cl_float3 m[3];
    } clTransform = {{
        { transform[0][0], transform[0][1], transform[0][2] },
        { transform[1][0], transform[1][1], transform[1][2] },
        { transform[2][0], transform[2][1], transform[2][2] }
    }};

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // linearImage
                                    cl::Image2D,  // ltmMaskImage
                                    cl::Image2D,  // rgbImage
                                    Matrix3x3,    // transform
                                    RGBConversionParameters    // demosaicParameters
                                    >(program, "convertTosRGB");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbImage->width, rgbImage->height),
           linearImage.getImage2D(), ltmMaskImage.getImage2D(), rgbImage->getImage2D(), clTransform, demosaicParameters.rgbConversionParameters);
}

void convertToGrayscale(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                        gls::cl_image_2d<float>* grayscaleImage,
                        const DemosaicParameters& demosaicParameters) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    const auto& transform = demosaicParameters.rgb_cam;

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // linearImage
                                    cl::Image2D,  // grayscaleImage
                                    cl_float3     // transform
                                    >(program, "convertToGrayscale");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(grayscaleImage->width, grayscaleImage->height),
           linearImage.getImage2D(), grayscaleImage->getImage2D(), { transform[0][0], transform[0][1], transform[0][2] });
}

void despeckleImage(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                    const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                    gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl_float3,    // var_a
                                    cl_float3,    // var_b
                                    cl::Image2D   // outputImage
                                    >(program, "despeckleLumaMedianChromaImage");

    cl_float3 cl_var_a = { var_a[0], var_a[1], var_a[2] };
    cl_float3 cl_var_b = { var_b[0], var_b[1], var_b[2] };

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), cl_var_a, cl_var_b, outputImage->getImage2D());
}

// --- Multiscale Noise Reduction ---
// https://www.cns.nyu.edu/pub/lcv/rajashekar08a.pdf

void denoiseImage(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                  const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                  const gls::Vector<3> thresholdMultipliers,
                  float chromaBoost, float gradientBoost, float gradientThreshold,
                  gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl::Image2D,  // gradientImage
                                    cl_float3,    // var_a
                                    cl_float3,    // var_b
                                    cl_float3,    // thresholdMultipliers
                                    float,        // chromaBoost
                                    float,        // gradientBoost
                                    float,        // gradientThreshold
                                    cl::Image2D   // outputImage
                                    >(program, "denoiseImage");

    cl_float3 cl_var_a = { var_a[0], var_a[1], var_a[2] };
    cl_float3 cl_var_b = { var_b[0], var_b[1], var_b[2] };

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), gradientImage.getImage2D(),
           cl_var_a, cl_var_b, { thresholdMultipliers[0], thresholdMultipliers[1], thresholdMultipliers[2] },
           chromaBoost, gradientBoost, gradientThreshold, outputImage->getImage2D());
}

void denoiseImageGuided(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                        const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                        gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl_float3,    // var_a
                                    cl_float3,    // ver_b
                                    cl::Image2D   // outputImage
                                    >(program, "denoiseImageGuided");

    cl_float3 cl_var_a = { var_a[0], var_a[1], var_a[2] };
    cl_float3 cl_var_b = { var_b[0], var_b[1], var_b[2] };

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), cl_var_a, cl_var_b, outputImage->getImage2D());
}

// Arrays order is LF, MF, HF
void localToneMappingMask(gls::OpenCLContext* glsContext,
                          const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                          const std::array<const gls::cl_image_2d<gls::rgba_pixel_float>*, 3>& guideImage,
                          const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abImage,
                          const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abMeanImage,
                          const LTMParameters& ltmParameters,
                          const gls::Matrix<3, 3>& ycbcr_srgb,
                          const gls::Vector<2>& nlf,
                          gls::cl_image_2d<gls::luma_pixel_float>* outputImage) {
    for (int i = 0; i < 3; i++) {
        assert(guideImage[i]->width == abImage[i]->width && guideImage[i]->height == abImage[i]->height);
        assert(guideImage[i]->width == abMeanImage[i]->width && guideImage[i]->height == abMeanImage[i]->height);
    }

    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    struct Matrix3x3 {
        cl_float3 m[3];
    } cl_ycbcr_srgb = {{
        { ycbcr_srgb[0][0], ycbcr_srgb[0][1], ycbcr_srgb[0][2] },
        { ycbcr_srgb[1][0], ycbcr_srgb[1][1], ycbcr_srgb[1][2] },
        { ycbcr_srgb[2][0], ycbcr_srgb[2][1], ycbcr_srgb[2][2] }
    }};

    const auto linear_sampler = cl::Sampler(glsContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

    // Bind the kernel parameters
    auto gfKernel = cl::KernelFunctor<cl::Image2D,  // guideImage
                                      cl::Image2D,  // abImage
                                      float,        // eps
                                      cl::Sampler   // linear_sampler
                                      >(program, "GuidedFilterABImage");

    auto gfMeanKernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                          cl::Image2D,  // outputImage
                                          cl::Sampler   // linear_sampler
                                          >(program, "BoxFilterGFImage");

    auto ltmKernel = cl::KernelFunctor<cl::Image2D,     // inputImage
                                       cl::Image2D,     // lfAbImage
                                       cl::Image2D,     // mfAbImage
                                       cl::Image2D,     // hfAbImage
                                       cl::Image2D,     // ltmMaskImage
                                       LTMParameters,   // ltmParameters
                                       Matrix3x3,       // ycbcr_srgb
                                       cl_float2,       // nlf
                                       cl::Sampler      // linear_sampler
                                       >(program, "localToneMappingMaskImage");

    try {
        // Schedule the kernel on the GPU
        for (int i = 0; i < 3; i++) {
            if (i == 0 || ltmParameters.detail[i] != 1) {
                gfKernel(gls::OpenCLContext::buildEnqueueArgs(guideImage[i]->width, guideImage[i]->height),
                         guideImage[i]->getImage2D(), abImage[i]->getImage2D(), ltmParameters.eps, linear_sampler);

                gfMeanKernel(gls::OpenCLContext::buildEnqueueArgs(abImage[i]->width, abImage[i]->height),
                             abImage[i]->getImage2D(), abMeanImage[i]->getImage2D(), linear_sampler);
            }
        }

        ltmKernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
                  inputImage.getImage2D(), abMeanImage[0]->getImage2D(), abMeanImage[1]->getImage2D(), abMeanImage[2]->getImage2D(), outputImage->getImage2D(),
                  ltmParameters, cl_ycbcr_srgb, { nlf[0], nlf[1] }, linear_sampler);
    } catch (cl::Error& e) {
        std::cout << "Error: " << e.what() << " - " << gls::clStatusToString(e.err()) << std::endl;
        throw std::logic_error("This is bad");
    }
}

void bayerToRawRGBA(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                    gls::cl_image_2d<gls::rgba_pixel_float>* rgbaImage,
                    BayerPattern bayerPattern) {
    assert(rawImage.width == 2 * rgbaImage->width && rawImage.height == 2 * rgbaImage->height);

    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rawImage
                                    cl::Image2D,  // rgbaImage
                                    int           // bayerPattern
                                    >(program, "bayerToRawRGBA");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbaImage->width, rgbaImage->height),
           rawImage.getImage2D(), rgbaImage->getImage2D(), bayerPattern);
}

void rawRGBAToBayer(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& rgbaImage,
                    gls::cl_image_2d<gls::luma_pixel_float>* rawImage,
                    BayerPattern bayerPattern) {
    assert(rawImage->width == 2 * rgbaImage.width && rawImage->height == 2 * rgbaImage.height);

    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // rgbaImage
                                    cl::Image2D,  // rawImage
                                    int           // bayerPattern
                                    >(program, "rawRGBAToBayer");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(rgbaImage.width, rgbaImage.height),
           rgbaImage.getImage2D(), rawImage->getImage2D(), bayerPattern);
}

void denoiseRawRGBAImage(gls::OpenCLContext* glsContext,
                         const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                         const gls::Vector<4> rawVariance,
                         gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl_float4,    // rawVariance
                                    cl::Image2D   // outputImage
                                    >(program, "denoiseRawRGBAImage");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), { rawVariance[0], rawVariance[1], rawVariance[2], rawVariance[3] }, outputImage->getImage2D());
}

void despeckleRawRGBAImage(gls::OpenCLContext* glsContext,
                           const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                           const gls::Vector<4> rawVariance,
                           gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl_float4,    // rawVariance
                                    cl::Image2D   // outputImage
                                    >(program, "despeckleRawRGBAImage");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), { rawVariance[0], rawVariance[1], rawVariance[2], rawVariance[3] }, outputImage->getImage2D());
}

void gaussianBlurImage(gls::OpenCLContext* glsContext,
                       const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                       float radius,
                       gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    const bool ordinary_gaussian = false;
    if (ordinary_gaussian) {
        // Bind the kernel parameters
        auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                        float,        // radius
                                        cl::Image2D   // outputImage
                                        >(program, "gaussianBlurImage");

        // Schedule the kernel on the GPU
        kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
               inputImage.getImage2D(), radius, outputImage->getImage2D());
    } else {
        const int kernelSize = (int) (2 * ceil(2 * radius) + 1);

        std::vector<float> weights(kernelSize*kernelSize);
        for (int y = -kernelSize / 2, i = 0; y <= kernelSize / 2; y++) {
            for (int x = -kernelSize / 2; x <= kernelSize / 2; x++, i++) {
                weights[i] = exp(-((float)(x * x + y * y) / (2 * radius * radius)));
            }
        }
        std::cout << "Gaussian Kernel weights (" << weights.size() << "): " << std::endl;
        for (const auto& w : weights) {
            std::cout << std::setprecision(4) << std::scientific << w << ", ";
        }
        std::cout << std::endl;

        const int outWidth = kernelSize / 2 + 1;
        const int weightsCount = outWidth * outWidth;
        std::vector<std::tuple<float, float, float>> weightsOut(weightsCount);
        KernelOptimizeBilinear2d(kernelSize, weights, &weightsOut);

        std::cout << "Bilinear Gaussian Kernel weights and offsets (" << weightsOut.size() << "): " << std::endl;
        for (const auto& [w, x, y] : weightsOut) {
            std::cout << w << " @ (" << x << " : " << y << "), " << std::endl;
        }

        cl::Buffer weightsBuffer(weightsOut.begin(), weightsOut.end(), /* readOnly */ true, /* useHostPtr */ false);

        const auto linear_sampler = cl::Sampler(glsContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

        // Bind the kernel parameters
        auto kernel = cl::KernelFunctor<cl::Image2D,    // inputImage
                                        int,            // samples
                                        cl::Buffer,     // weights
                                        cl::Image2D,    // outputImage
                                        cl::Sampler     // linear_sampler
                                        >(program, "sampledConvolution");

        // Schedule the kernel on the GPU
        kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
               inputImage.getImage2D(), weightsCount, weightsBuffer, outputImage->getImage2D(), linear_sampler);
    }
}

void blueNoiseImage(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                    const gls::cl_image_2d<gls::luma_pixel_16>& blueNoiseImage,
                    gls::Vector<2> lumaVariance,
                    gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl::Image2D,  // blueNoiseImage
                                    cl_float2,    // lumaVariance
                                    cl::Image2D,  // outputImage
                                    cl::Sampler   // linear_sampler
                                    >(program, "blueNoiseImage");

    const auto linear_sampler = cl::Sampler(glsContext->clContext(), true, CL_ADDRESS_REPEAT, CL_FILTER_LINEAR);

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), blueNoiseImage.getImage2D(),
           { lumaVariance[0], lumaVariance[1] },
           outputImage->getImage2D(), linear_sampler);
}


void blendHighlightsImage(gls::OpenCLContext* glsContext,
                          const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                          float clip,
                          gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    float,        // clip
                                    cl::Image2D   // outputImage
                                    >(program, "blendHighlightsImage");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), clip, outputImage->getImage2D());
}

YCbCrNLF MeasureYCbCrNLF(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel_float>& image, float exposure_multiplier) {
    gls::cl_image_2d<gls::rgba_pixel_float> noiseStats(glsContext->clContext(), image.width, image.height);
    applyKernel(glsContext, "noiseStatistics", image, &noiseStats);
    const auto noiseStatsCpu = noiseStats.mapImage();

    // Only consider pixels with variance lower than the expected noise value
    double varianceMax = 0.001;

    // Limit to pixels the more linear intensity zone of the sensor
    const double maxValue = 0.5;
    const double minValue = 0.001;

    // Collect pixel statistics
    double s_x = 0;
    double s_xx = 0;
    gls::DVector<3> s_y = gls::DVector<3>::zeros();
    gls::DVector<3> s_xy = gls::DVector<3>::zeros();

    double N = 0;
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = { ns[1], ns[2], ns[3] };

        if (m >= minValue && m <= maxValue && all(v <= gls::DVector<3>(varianceMax))) {
            s_x += m;
            s_y += v;
            s_xx += m * m;
            s_xy += m * v;
            N++;
        }
    });

    // Linear regression on pixel statistics to extract a linear noise model: nlf = A + B * Y
    auto nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    auto nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    // Estimate regression mean square error
    gls::DVector<3> err2 = gls::DVector<3>::zeros();
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = { ns[1], ns[2], ns[3] };

        if (m >= minValue && m <= maxValue && all(v <= gls::DVector<3>(varianceMax))) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            err2 += diff * diff;
        }
    });
    err2 /= N;

//    std::cout << "1) Pyramid NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(err2)
//              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels"<< std::endl;

    // Redo the statistics collection limiting the sample to pixels that fit well the linear model
    s_x = 0;
    s_xx = 0;
    s_y = gls::DVector<3>::zeros();
    s_xy = gls::DVector<3>::zeros();
    N = 0;
    gls::DVector<3> newErr2 = gls::DVector<3>::zeros();
    int discarded = 0;
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = { ns[1], ns[2], ns[3] };

        if (m >= minValue && m <= maxValue && all(v <= gls::DVector<3>(varianceMax))) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = abs(nlfP - v);
            auto diffSquare = diff * diff;

            if (all(diffSquare <= 0.5 * err2)) {
                s_x += m;
                s_y += v;
                s_xx += m * m;
                s_xy += m * v;
                N++;
                newErr2 += diffSquare;
            } else {
                discarded++;
            }
        }
    });
    newErr2 /= N;

    // Estimate the new regression parameters
    nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    assert(all(newErr2 < err2));

    std::cout << "Pyramid NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(newErr2)
              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels" << std::endl;

    noiseStats.unmapImage(noiseStatsCpu);

    double varianceExposureAdjustment = exposure_multiplier * exposure_multiplier;

    nlfA *= varianceExposureAdjustment;
    nlfB *= varianceExposureAdjustment;

    return std::pair (
        gls::Vector<3> { (float) nlfA[0], (float) nlfA[1], (float) nlfA[2] }, // A values
        gls::Vector<3> { (float) nlfB[0], (float) nlfB[1], (float) nlfB[2] }  // B values
    );
}

RawNLF MeasureRawNLF(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::luma_pixel_float>& rawImage, float exposure_multiplier, BayerPattern bayerPattern) {
    gls::cl_image_2d<gls::rgba_pixel_float> meanImage(glsContext->clContext(), rawImage.width / 2, rawImage.height / 2);
    gls::cl_image_2d<gls::rgba_pixel_float> varImage(glsContext->clContext(), rawImage.width / 2, rawImage.height / 2);

    rawNoiseStatistics(glsContext, rawImage, bayerPattern, &meanImage, &varImage);

    const auto meanImageCpu = meanImage.mapImage();
    const auto varImageCpu = varImage.mapImage();

    // Only consider pixels with variance lower than the expected noise value
    double varianceMax = 0.001;

    // Limit to pixels the more linear intensity zone of the sensor
    const double maxValue = 0.5;
    const double minValue = 0.001;

    // Collect pixel statistics
    gls::DVector<4> s_x = gls::DVector<4>::zeros();
    gls::DVector<4> s_y = gls::DVector<4>::zeros();
    gls::DVector<4> s_xx = gls::DVector<4>::zeros();
    gls::DVector<4> s_xy = gls::DVector<4>::zeros();

    double N = 0;
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = { mm[0], mm[1], mm[2], mm[3] };
        gls::DVector<4> v = { vv[0], vv[1], vv[2], vv[3] };

        if (all(m >= gls::DVector<4>(minValue)) && all(v <= gls::DVector<4>(maxValue)) && all(v <= gls::DVector<4>(varianceMax))) {
            s_x += m;
            s_y += v;
            s_xx += m * m;
            s_xy += m * v;
            N++;
        }
    });

    // Linear regression on pixel statistics to extract a linear noise model: nlf = A + B * Y
    auto nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    auto nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    // Estimate regression mean square error
    gls::DVector<4> err2 = gls::DVector<4>::zeros();
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = { mm[0], mm[1], mm[2], mm[3] };
        gls::DVector<4> v = { vv[0], vv[1], vv[2], vv[3] };

        if (all(m >= gls::DVector<4>(minValue)) && all(v <= gls::DVector<4>(maxValue)) && all(v <= gls::DVector<4>(varianceMax))) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            err2 += diff * diff;
        }
    });
    err2 /= N;

//    std::cout << "RAW NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(err2)
//              << " on " << std::setprecision(1) << std::fixed << 100 * N / (rawImage.width * rawImage.height) << "% pixels"<< std::endl;

    // Redo the statistics collection limiting the sample to pixels that fit well the linear model
    s_x = gls::DVector<4>::zeros();
    s_y = gls::DVector<4>::zeros();
    s_xx = gls::DVector<4>::zeros();
    s_xy = gls::DVector<4>::zeros();
    N = 0;
    gls::DVector<4> newErr2 = gls::DVector<4>::zeros();
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = { mm[0], mm[1], mm[2], mm[3] };
        gls::DVector<4> v = { vv[0], vv[1], vv[2], vv[3] };

        if (all(m >= gls::DVector<4>(minValue)) && all(v <= gls::DVector<4>(maxValue)) && all(v <= gls::DVector<4>(varianceMax))) {
            const auto nlfP = nlfA + nlfB * m;
            const auto diff = abs(nlfP - v);
            const auto diffSquare = diff * diff;

            if (all(diffSquare <= 0.5 * err2)) {
                s_x += m;
                s_y += v;
                s_xx += m * m;
                s_xy += m * v;
                N++;
                newErr2 += diffSquare;
            }
        }
    });
    newErr2 /= N;

    // Estimate the new regression parameters
    nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    assert(all(newErr2 < err2));

    std::cout << "RAW NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(newErr2)
              << " on " << std::setprecision(1) << std::fixed << 100 * N / (rawImage.width * rawImage.height) << "% pixels"<< std::endl;

    meanImage.unmapImage(meanImageCpu);
    varImage.unmapImage(varImageCpu);

    double varianceExposureAdjustment = exposure_multiplier * exposure_multiplier;

    nlfA *= varianceExposureAdjustment;
    nlfB *= varianceExposureAdjustment;

    return std::pair (
        gls::Vector<4> { (float) nlfA[0], (float) nlfA[1], (float) nlfA[2], (float) nlfA[3] }, // A values
        gls::Vector<4> { (float) nlfB[0], (float) nlfB[1], (float) nlfB[2], (float) nlfB[3] }  // B values
    );
}

void clFuseFrames(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& referenceImage,
                  const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& previousFusedImage,
                  const gls::Matrix<3, 3>& homography,
                  const gls::Vector<3>& var_a, const gls::Vector<3>& var_b, int fusedFrames,
                  gls::cl_image_2d<gls::rgba_pixel_float>* newFusedImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,        // referenceImage
                                    cl::Image2D,        // gradientImage
                                    cl::Image2D,        // inputImage
                                    cl::Image2D,        // inputFusedImage
                                    gls::Matrix<3, 3>,  // homography
                                    cl::Sampler,        // linear_sampler
                                    cl_float3,          // var_a
                                    cl_float3,          // var_b
                                    int,                // fusedFrames
                                    cl::Image2D         // outputFusedImage
                                    >(program, "fuseFrames");

    cl_float3 cl_var_a = { var_a[0], var_a[1], var_a[2] };
    cl_float3 cl_var_b = { var_b[0], var_b[1], var_b[2] };

    const auto linear_sampler = cl::Sampler(glsContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(newFusedImage->width, newFusedImage->height),
           referenceImage.getImage2D(),
           gradientImage.getImage2D(),
           inputImage.getImage2D(),
           previousFusedImage.getImage2D(),
           homography, linear_sampler,
           cl_var_a, cl_var_b,
           fusedFrames,
           newFusedImage->getImage2D());
}

template <typename T>
void subtractNoiseFusedImage(gls::OpenCLContext* glsContext,
                             const gls::cl_image_2d<T>& inputImage,
                             const gls::cl_image_2d<T>& inputImage1,
                             const gls::cl_image_2d<T>& inputImageDenoised1,
                             gls::cl_image_2d<T>* outputImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("demosaic");

    const auto linear_sampler = cl::Sampler(glsContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,  // inputImage
                                    cl::Image2D,  // inputImage1
                                    cl::Image2D,  // inputImageDenoised1
                                    cl::Image2D,  // outputImage
                                    cl::Sampler   // linear_sampler
                                    >(program, "subtractNoiseFusedImage");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
           inputImage.getImage2D(), inputImage1.getImage2D(), inputImageDenoised1.getImage2D(),
           outputImage->getImage2D(), linear_sampler);
}

template
void subtractNoiseFusedImage(gls::OpenCLContext* glsContext,
                             const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                             const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage1,
                             const gls::cl_image_2d<gls::rgba_pixel_float>& inputImageDenoised1,
                             gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

template <typename T>
void clRescaleImage(gls::OpenCLContext* cLContext,
                    const gls::cl_image_2d<T>& inputImage,
                    gls::cl_image_2d<T>* outputImage) {
    // Load the shader source
    const auto program = cLContext->loadProgram("demosaic");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,        // inputImage
                                    cl::Image2D,        // outputImage
                                    cl::Sampler         // linear_sampler
                                    >(program, "rescaleImage");

    const auto linear_sampler = cl::Sampler(cLContext->clContext(), true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR);

    kernel(
#if __APPLE__
           gls::OpenCLContext::buildEnqueueArgs(outputImage->width, outputImage->height),
#else
           cl::EnqueueArgs(cl::NDRange(outputImage->width, outputImage->height), cl::NDRange(32, 32)),
#endif
           inputImage.getImage2D(), outputImage->getImage2D(), linear_sampler);
}

template
void clRescaleImage(gls::OpenCLContext* cLContext,
                    const gls::cl_image_2d<gls::rgba_pixel>& inputImage,
                    gls::cl_image_2d<gls::rgba_pixel>* outputImage);

template
void clRescaleImage(gls::OpenCLContext* cLContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                    gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);
