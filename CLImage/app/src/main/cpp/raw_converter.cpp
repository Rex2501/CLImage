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

#include "raw_converter.hpp"

#include "gls_logging.h"

static const char* TAG = "RAW Converter";

#define PRINT_EXECUTION_TIME true

void RawConverter::allocateTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!clRawImage) {
        clRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_16>>(clContext, width, height);
        clScaledRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
        clGreenImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
        clLinearRGBImageA = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width, height);
        clLinearRGBImageB = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width, height);
        clsRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel>>(clContext, width, height);

        // Placeholder, only allocated if LTM is used
        ltmMaskImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, 1, 1);

        pyramidalDenoise = std::make_unique<PyramidalDenoise<5>>(glsContext, width, height);
    }
}

void RawConverter::allocateLtmMaskImage(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (ltmMaskImage->width != width || ltmMaskImage->height != height) {
        ltmMaskImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
        ltmLFAbGfImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/16, height/16);
        ltmMeanLFAbGfImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/16, height/16);
        ltmMFAbGfImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/4, height/4);
        ltmMeanMFAbGfImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/4, height/4);
        ltmHFAbGfImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width, height);
        ltmMeanHFAbGfImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width, height);
    }
}

void RawConverter::allocateHighNoiseTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!rgbaRawImage) {
        rgbaRawImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
        denoisedRgbaRawImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
    }
}

void RawConverter::allocateFastDemosaicTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!clFastLinearRGBImage) {
        clRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_16>>(clContext, width, height);
        clScaledRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
        clFastLinearRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
        clsFastRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel>>(clContext, width/2, height/2);

        // Placeholder, not used in Fast Demosaic
        ltmMaskImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, 1, 1);
    }
}

gls::cl_image_2d<gls::rgba_pixel>* RawConverter::demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                               DemosaicParameters* demosaicParameters,
                                                               const gls::rectangle* gmb_position, bool rotate_180) {
    auto clContext = _glsContext->clContext();

    LOG_INFO(TAG) << "Begin Demosaicing..." << std::endl;

    NoiseModel* noiseModel = &demosaicParameters->noiseModel;

    LOG_INFO(TAG) << "NoiseLevel: " << demosaicParameters->noiseLevel << std::endl;

    allocateTextures(_glsContext, rawImage.width, rawImage.height);

    if (demosaicParameters->rgbConversionParameters.localToneMapping) {
        allocateLtmMaskImage(_glsContext, rawImage.width, rawImage.height);
    }

#ifdef PRINT_EXECUTION_TIME
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    // Copy input data to the OpenCL input buffer
    clRawImage->copyPixelsFrom(rawImage);

    // --- Image Demosaicing ---

    scaleRawData(_glsContext, *clRawImage, clScaledRawImage.get(), demosaicParameters->bayerPattern, demosaicParameters->scale_mul,
                 demosaicParameters->black_level / 0xffff);

    const auto rawNLF = computeRawNoiseStatistics(_glsContext, *clScaledRawImage, demosaicParameters->bayerPattern);

    demosaicParameters->noiseModel.rawNlf = gls::Vector<4> { rawNLF[4], rawNLF[5], rawNLF[6], rawNLF[7] };

    const bool high_noise_image = (demosaicParameters->noiseModel.rawNlf[1] + demosaicParameters->noiseModel.rawNlf[3]) / 2 > 1e-03;

    if (high_noise_image) {
        allocateHighNoiseTextures(_glsContext, rawImage.width, rawImage.height);
    }

    if (high_noise_image) {
        std::cout << "denoiseRawRGBAImage" << std::endl;

        bayerToRawRGBA(_glsContext, *clScaledRawImage, rgbaRawImage.get(), demosaicParameters->bayerPattern);

        despeckleRawRGBAImage(_glsContext, *rgbaRawImage, demosaicParameters->noiseModel.rawNlf, denoisedRgbaRawImage.get());

        // denoiseRawRGBAImage(_glsContext, *denoisedRgbaRawImage, noiseModel->rawNlf, rgbaRawImage.get());

        // applyKernel(_glsContext, "medianFilterImage3x3x4", *rgbaRawImage, denoisedRgbaRawImage.get());

        rawRGBAToBayer(_glsContext, *denoisedRgbaRawImage, clScaledRawImage.get(), demosaicParameters->bayerPattern);
    }

    gls::Vector<2> greenVariance = { (rawNLF[1] + rawNLF[3]) / 2, (rawNLF[5] + rawNLF[7]) / 2 };
    gls::Vector<2> redVariance = { rawNLF[0], rawNLF[4] };
    gls::Vector<2> blueVariance = { rawNLF[3], rawNLF[6] };

    // TODO: why divide by 4?
    interpolateGreen(_glsContext, *clScaledRawImage, clGreenImage.get(), demosaicParameters->bayerPattern, greenVariance);

    interpolateRedBlue(_glsContext, *clScaledRawImage, *clGreenImage, clLinearRGBImageA.get(), demosaicParameters->bayerPattern,
                       redVariance, blueVariance, rotate_180);

    // Recover clipped highlights
    blendHighlightsImage(_glsContext, *clLinearRGBImageA, /*clip=*/ 1.0, clLinearRGBImageA.get());

    // --- Image Denoising ---

    // Convert linear image to YCbCr
    auto cam_to_ycbcr = cam_ycbcr(demosaicParameters->rgb_cam);

    std::cout << "cam_to_ycbcr: " << cam_to_ycbcr.span() << std::endl;

    transformImage(_glsContext, *clLinearRGBImageA, clLinearRGBImageA.get(), cam_to_ycbcr);

    // Luma and Chroma Despeckling
    std::cout << "despeckleImage" << std::endl;
    const auto& np = noiseModel->pyramidNlf[0];
    despeckleImage(_glsContext, *clLinearRGBImageA, { np[0], np[1], np[2] }, { np[3], np[4], np[5] }, clLinearRGBImageB.get());

    // False Color Removal
    // TODO: this is expensive, make it an optional stage
//    std::cout << "falseColorsRemovalImage" << std::endl;
//    applyKernel(_glsContext, "falseColorsRemovalImage", *clLinearRGBImageB, clLinearRGBImageA.get());

    const auto normalized_ycbcr_to_cam = inverse(cam_to_ycbcr) * demosaicParameters->exposure_multiplier;

    auto clDenoisedImage = pyramidalDenoise->denoise(_glsContext, &(demosaicParameters->denoiseParameters),
                                                     clLinearRGBImageB.get(),
                                                     demosaicParameters->rgb_cam, gmb_position, rotate_180,
                                                     &(noiseModel->pyramidNlf));

    std::cout << "pyramidNlf:\n" << std::scientific << noiseModel->pyramidNlf << std::endl;

    if (demosaicParameters->rgbConversionParameters.localToneMapping) {
        const std::array<const gls::cl_image_2d<gls::rgba_pixel_float>*, 3>& guideImage = {
            pyramidalDenoise->denoisedImagePyramid[4].get(),
            pyramidalDenoise->denoisedImagePyramid[2].get(),
            pyramidalDenoise->denoisedImagePyramid[0].get()
        };
        const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abImage = {
            ltmLFAbGfImage.get(), ltmMFAbGfImage.get(), ltmHFAbGfImage.get()
        };
        const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abMeanImage = {
            ltmMeanLFAbGfImage.get(), ltmMeanMFAbGfImage.get(), ltmMeanHFAbGfImage.get()
        };

        gls::Vector<2> nlf = { noiseModel->pyramidNlf[0][0], noiseModel->pyramidNlf[0][3] };
        localToneMappingMask(_glsContext, *clDenoisedImage, guideImage, abImage, abMeanImage, demosaicParameters->ltmParameters, ycbcr_srgb, nlf, ltmMaskImage.get());
    }

    // Convert result back to camera RGB
    transformImage(_glsContext, *clDenoisedImage, clDenoisedImage, normalized_ycbcr_to_cam);

    // --- Image Post Processing ---

    convertTosRGB(_glsContext, *clDenoisedImage, *ltmMaskImage, clsRGBImage.get(), *demosaicParameters);

#ifdef PRINT_EXECUTION_TIME
    cl::CommandQueue queue = cl::CommandQueue::getDefault();
    queue.finish();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;
#endif

    return clsRGBImage.get();
}

gls::cl_image_2d<gls::rgba_pixel>* RawConverter::fastDemosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                                   const DemosaicParameters& demosaicParameters) {
    allocateFastDemosaicTextures(_glsContext, rawImage.width, rawImage.height);

    LOG_INFO(TAG) << "Begin Fast Demosaicing (GPU)..." << std::endl;

#ifdef PRINT_EXECUTION_TIME
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    // Copy input data to the OpenCL input buffer
    clRawImage->copyPixelsFrom(rawImage);

    // --- Image Demosaicing ---

    scaleRawData(_glsContext, *clRawImage, clScaledRawImage.get(), demosaicParameters.bayerPattern, demosaicParameters.scale_mul,
                 demosaicParameters.black_level / 0xffff);

    fasteDebayer(_glsContext, *clScaledRawImage, clFastLinearRGBImage.get(), demosaicParameters.bayerPattern);

    // Recover clipped highlights
    blendHighlightsImage(_glsContext, *clFastLinearRGBImage, /*clip=*/ 1.0, clFastLinearRGBImage.get());

    // --- Image Post Processing ---

    convertTosRGB(_glsContext, *clFastLinearRGBImage, *ltmMaskImage, clsFastRGBImage.get(), demosaicParameters);

#ifdef PRINT_EXECUTION_TIME
    cl::CommandQueue queue = cl::CommandQueue::getDefault();
    queue.finish();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;
#endif

    return clsFastRGBImage.get();
}

/*static*/ gls::image<gls::rgb_pixel>::unique_ptr RawConverter::convertToRGBImage(const gls::cl_image_2d<gls::rgba_pixel>& clRGBAImage) {
    auto rgbImage = std::make_unique<gls::image<gls::rgb_pixel>>(clRGBAImage.width, clRGBAImage.height);
    auto rgbaImage = clRGBAImage.mapImage();
    for (int y = 0; y < clRGBAImage.height; y++) {
        for (int x = 0; x < clRGBAImage.width; x++) {
            const auto& p = rgbaImage[y][x];
            (*rgbImage)[y][x] = { p.red, p.green, p.blue };
        }
    }
    clRGBAImage.unmapImage(rgbaImage);
    return rgbImage;
}
