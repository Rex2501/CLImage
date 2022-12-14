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

#ifndef pyramidal_denoise_h
#define pyramidal_denoise_h

#include "demosaic.hpp"
#include "demosaic_cl.hpp"

template <size_t levels>
struct PyramidProcessor {
    const int width, height;
    int fusedFrames;

    typedef gls::cl_image_2d<gls::rgba_pixel_float> imageType;
    std::array<imageType::unique_ptr, levels-1> imagePyramid;
    std::array<gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr, levels-1> gradientPyramid;
    std::array<imageType::unique_ptr, levels> subtractedImagePyramid;
    std::array<imageType::unique_ptr, levels> denoisedImagePyramid;
    std::array<imageType::unique_ptr, levels> fusionImagePyramidA;
    std::array<imageType::unique_ptr, levels> fusionImagePyramidB;
    std::array<imageType::unique_ptr, levels> fusionReferenceImagePyramid;
    std::array<gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr, levels> fusionReferenceGradientPyramid;
    std::array<imageType::unique_ptr, levels>* fusionBuffer[2];

    PyramidProcessor(gls::OpenCLContext* glsContext, int width, int height);

    imageType* denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, levels>* denoiseParameters,
                       const imageType& image,
                       const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                       std::array<YCbCrNLF, levels>* nlfParameters,
                       float exposure_multiplier, bool calibrateFromImage = false);

    void fuseFrame(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, levels>* denoiseParameters,
                   const imageType& image,
                   const gls::Matrix<3, 3>& homography,
                   const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                   std::array<YCbCrNLF, levels>* nlfParameters,
                   float exposure_multiplier, bool calibrateFromImage = false);

    imageType* getFusedImage(gls::OpenCLContext* glsContext);
};

#endif /* pyramidal_denoise_h */
