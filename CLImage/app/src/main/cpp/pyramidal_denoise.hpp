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

#include "guided_filter.hpp"

struct ImageDenoiser {
    ImageDenoiser(gls::OpenCLContext* glsContext, int width, int height) {}

    virtual ~ImageDenoiser() { }

    virtual void denoise(gls::OpenCLContext* glsContext,
                         const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                         const gls::Vector<3>& sigma_a, const gls::Vector<3>& sigma_b,
                         float chromaBoost, float gradientBoost, int pyramidLevel,
                         gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) = 0;
};

template <size_t levels>
struct PyramidalDenoise {
    enum DenoiseAlgorithm { Bilateral, GuidedFast, GuidedPrecise };

    typedef gls::cl_image_2d<gls::rgba_pixel_float> imageType;
    std::array<imageType::unique_ptr, levels-1> imagePyramid;
    std::array<imageType::unique_ptr, levels> denoisedImagePyramid;
    std::array<std::unique_ptr<ImageDenoiser>, levels> denoiser;

    PyramidalDenoise(gls::OpenCLContext* glsContext, int width, int height, DenoiseAlgorithm _denoiseAlgorithm = Bilateral);

    imageType* denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, levels>* denoiseParameters,
                       imageType* image, const gls::Matrix<3, 3>& rgb_cam,
                       const gls::rectangle* gmb_position, bool rotate_180,
                       gls::Matrix<levels, 6>* nlfParameters);
};

gls::Vector<6> computeNoiseStatistics(gls::OpenCLContext* glsContext,
                                      const gls::cl_image_2d<gls::rgba_pixel_float>& image);

gls::Vector<8> computeRawNoiseStatistics(gls::OpenCLContext* glsContext,
                                         const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                                         BayerPattern bayerPattern);

#endif /* pyramidal_denoise_h */
