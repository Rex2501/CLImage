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

#ifndef demosaic_cl_hpp
#define demosaic_cl_hpp

#include "demosaic.hpp"

#include "gls_cl_image.hpp"

template <typename T1, typename T2>
void applyKernel(gls::OpenCLContext* glsContext, const std::string& kernelName,
                 const gls::cl_image_2d<T1>& inputImage,
                 gls::cl_image_2d<T2>* outputImage);

void scaleRawData(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::luma_pixel_16>& rawImage,
                  gls::cl_image_2d<gls::luma_pixel_float>* scaledRawImage,
                  BayerPattern bayerPattern, gls::Vector<4> scaleMul, float blackLevel);

void rawImageGradient(gls::OpenCLContext* glsContext,
                      const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                      gls::cl_image_2d<gls::luma_alpha_pixel_float>* gradientImage);

void rawImageSobel(gls::OpenCLContext* glsContext,
                   const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                   gls::cl_image_2d<gls::rgba_pixel_float>* gradientImage);

void interpolateGreen(gls::OpenCLContext* glsContext,
                      const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                      const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                      gls::cl_image_2d<gls::luma_pixel_float>* greenImage,
                      BayerPattern bayerPattern, gls::Vector<2> greenVariance);

void interpolateRedBlue(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                        const gls::cl_image_2d<gls::luma_pixel_float>& greenImage,
                        const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                        gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                        BayerPattern bayerPattern, gls::Vector<2> redVariance, gls::Vector<2> blueVariance);

void interpolateRedBlueAtGreen(gls::OpenCLContext* glsContext,
                               const gls::cl_image_2d<gls::rgba_pixel_float>& rgbImageIn,
                               const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                               gls::cl_image_2d<gls::rgba_pixel_float>* rgbImageOut,
                               BayerPattern bayerPattern, gls::Vector<2> redVariance, gls::Vector<2> blueVariance);

void malvar(gls::OpenCLContext* glsContext,
            const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
            const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
            gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
            BayerPattern bayerPattern, gls::Vector<2> redVariance,
            gls::Vector<2> greenVariance, gls::Vector<2> blueVariance);

void fasteDebayer(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                  gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                  BayerPattern bayerPattern);

void rawNoiseStatistics(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                        BayerPattern bayerPattern,
                        gls::cl_image_2d<gls::rgba_pixel_float>* meanImage,
                        gls::cl_image_2d<gls::rgba_pixel_float>* varImage);

template <typename T>
void resampleImage(gls::OpenCLContext* glsContext, const std::string& kernelName,
                   const gls::cl_image_2d<T>& inputImage, gls::cl_image_2d<T>* outputImage);

template <typename T>
void subtractNoiseImage(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<T>& inputImageDenoised0,
                        const gls::cl_image_2d<T>& inputImage1,
                        const gls::cl_image_2d<T>& inputImageDenoised1,
                        const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                        float luma_weight, float sharpening, const gls::Vector<2>& nlf, gls::cl_image_2d<T>* outputImage);

template <typename T>
void subtractNoiseFusedImage(gls::OpenCLContext* glsContext,
                             const gls::cl_image_2d<T>& inputImageDenoised0,
                             const gls::cl_image_2d<T>& inputImage1,
                             const gls::cl_image_2d<T>& inputImageDenoised1,
                             gls::cl_image_2d<T>* outputImage);

void transformImage(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                    gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                    const gls::Matrix<3, 3>& transform);

void convertTosRGB(gls::OpenCLContext* glsContext,
                   const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                   const gls::cl_image_2d<gls::luma_pixel_float>& ltmMaskImage,
                   gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                   const DemosaicParameters& demosaicParameters);

void convertToGrayscale(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                        gls::cl_image_2d<float>* grayscaleImage,
                        const DemosaicParameters& demosaicParameters);

void despeckleImage(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                    const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                    gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void denoiseImage(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                  const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                  const gls::Vector<3> thresholdMultipliers,
                  float chromaBoost, float gradientBoost, float gradientThreshold,
                  gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void denoiseImageGuided(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                        const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                        gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

// Arrays order is LF, MF, HF
void localToneMappingMask(gls::OpenCLContext* glsContext,
                          const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                          const std::array<const gls::cl_image_2d<gls::rgba_pixel_float>*, 3>& guideImage,
                          const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abImage,
                          const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abMeanImage,
                          const LTMParameters& ltmParameters,
                          const gls::Matrix<3, 3>& ycbcr_srgb,
                          const gls::Vector<2>& nlf,
                          gls::cl_image_2d<gls::luma_pixel_float>* outputImage);

void denoiseLumaImage(gls::OpenCLContext* glsContext,
                      const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                      const DenoiseParameters& denoiseParameters,
                      gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void bayerToRawRGBA(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                    gls::cl_image_2d<gls::rgba_pixel_float>* rgbaImage,
                    BayerPattern bayerPattern);

void rawRGBAToBayer(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& rgbaImage,
                    gls::cl_image_2d<gls::luma_pixel_float>* rawImage,
                    BayerPattern bayerPattern);

void denoiseRawRGBAImage(gls::OpenCLContext* glsContext,
                         const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                         const gls::Vector<4> rawVariance,
                         gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void despeckleRawRGBAImage(gls::OpenCLContext* glsContext,
                           const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                           const gls::Vector<4> rawVariance,
                           gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void gaussianBlurImage(gls::OpenCLContext* glsContext,
                       const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage, float radius,
                       gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void gaussianBlurSobelImage(gls::OpenCLContext* glsContext,
                            const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                            const gls::cl_image_2d<gls::rgba_pixel_float>& sobelImage,
                            std::array<float, 2> rawNoiseModel, float radius1, float radius2,
                            gls::cl_image_2d<gls::luma_alpha_pixel_float>* outputImage);

void blueNoiseImage(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                    const gls::cl_image_2d<gls::luma_pixel_16>& blueNoiseImage,
                    gls::Vector<2> lumaVariance,
                    gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

void blendHighlightsImage(gls::OpenCLContext* glsContext,
                          const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                          float clip,
                          gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

YCbCrNLF MeasureYCbCrNLF(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel_float>& image, float exposure_multiplier);

RawNLF MeasureRawNLF(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::luma_pixel_float>& rawImage, float exposure_multiplier, BayerPattern bayerPattern);

void clFuseFrames(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& referenceImage,
                  const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& previousFusedImage,
                  const gls::Matrix<3, 3>& homography,
                  const gls::Vector<3>& var_a, const gls::Vector<3>& var_b, int fusedFrames,
                  gls::cl_image_2d<gls::rgba_pixel_float>* newFusedImage);

template <typename T>
void clRescaleImage(gls::OpenCLContext* cLContext,
                    const gls::cl_image_2d<T>& inputImage,
                    gls::cl_image_2d<T>* outputImage);

#endif /* demosaic_cl_hpp */
