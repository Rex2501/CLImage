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

#include <iomanip>

#include "gls_logging.h"
#include "demosaic.hpp"

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

        pyramidProcessor = std::make_unique<PyramidProcessor<5>>(glsContext, width, height);

        // TODO: where do we keep the blue noise texture asset? Maybe generate this dynamically?
        const auto blueNoise = gls::image<gls::luma_pixel_16>::read_png_file("/Users/fabio/work/CLImage/CLImage/app/src/main/assets/HDR_L_0.png");
        clBlueNoise = std::make_unique<gls::cl_image_2d<gls::luma_pixel_16>>(_glsContext->clContext(), *blueNoise);
    }
}

void RawConverter::allocateHighNoiseTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!rgbaRawImage) {
        rgbaRawImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
        denoisedRgbaRawImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);

//        // TODO: where do we keep the blue noise texture asset? Maybe generate this dynamically?
//        const auto blueNoise = gls::image<gls::luma_pixel_16>::read_png_file("/Users/fabio/work/CLImage/CLImage/app/src/main/assets/HDR_L_0.png");
//        clBlueNoise = std::make_unique<gls::cl_image_2d<gls::luma_pixel_16>>(_glsContext->clContext(), *blueNoise);
    }
}

void RawConverter::allocateFastDemosaicTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!clFastLinearRGBImage) {
        clRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_16>>(clContext, width, height);
        clScaledRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
        clFastLinearRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
        clsFastRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel>>(clContext, width/2, height/2);
    }
}

template <typename T>
void SaveRawChannels(const gls::image<T>& rawImage, float maxVal, const std::string& basePath) {
    gls::image<gls::luma_pixel> chan0(rawImage.width/2, rawImage.height/2);
    gls::image<gls::luma_pixel> chan1(rawImage.width/2, rawImage.height/2);
    gls::image<gls::luma_pixel> chan2(rawImage.width/2, rawImage.height/2);
    gls::image<gls::luma_pixel> chan3(rawImage.width/2, rawImage.height/2);

    rawImage.apply([&chan0, &chan1, &chan2, &chan3, maxVal](const T& p, int x, int y){
        uint8_t val = 255 * std::sqrt(std::clamp((float) p, 0.0f, maxVal) / maxVal);

        if ((x & 0) == 0 && (y & 0) == 0) {
            chan0[y/2][x/2] = val;
        }
        if ((x & 1) == 0 && (y & 0) == 0) {
            chan1[y/2][x/2] = val;
        }
        if ((x & 0) == 0 && (y & 1) == 0) {
            chan2[y/2][x/2] = val;
        }
        if ((x & 1) == 0 && (y & 1) == 0) {
            chan3[y/2][x/2] = val;
        }
    });
    chan0.write_png_file(basePath + "0.png");
    chan1.write_png_file(basePath + "1.png");
    chan2.write_png_file(basePath + "2.png");
    chan3.write_png_file(basePath + "3.png");
}

void saveGreenChannel(const gls::cl_image_2d<gls::luma_pixel_float>& clGreenImage) {
    gls::image<gls::luma_pixel> out(clGreenImage.width, clGreenImage.height);
    const auto greenImageCPU = clGreenImage.mapImage();
    out.apply([&greenImageCPU](gls::luma_pixel* p, int x, int y){
        const auto& ip = greenImageCPU[y][x];
        *p = gls::luma_pixel {
            (uint8_t) (255 * std::sqrt(std::clamp((float) ip.luma, 0.0f, 1.0f)))
        };
    });
    clGreenImage.unmapImage(greenImageCPU);
    static int count = 1;
    out.write_png_file("/Users/fabio/green" + std::to_string(count++) + ".png");
}

std::array<gls::Vector<2>, 3> getRawVariance(const RawNLF& rawNLF) {
    const gls::Vector<2> greenVariance = { (rawNLF.first[1] + rawNLF.first[3]) / 2, (rawNLF.second[1] + rawNLF.second[3]) / 2 };
    const gls::Vector<2> redVariance = { rawNLF.first[0], rawNLF.second[0] };
    const gls::Vector<2> blueVariance = { rawNLF.first[2], rawNLF.second[2] };

    return { redVariance, greenVariance, blueVariance };
}

gls::cl_image_2d<gls::rgba_pixel_float>* RawConverter::demosaic(const gls::image<gls::luma_pixel_16>& rawImage,
                                                                DemosaicParameters* demosaicParameters, bool calibrateFromImage) {
    LOG_INFO(TAG) << "Begin Demosaicing..." << std::endl;

    allocateTextures(_glsContext, rawImage.width, rawImage.height);

    if (demosaicParameters->rgbConversionParameters.localToneMapping) {
        localToneMapping->allocateTextures(_glsContext, rawImage.width, rawImage.height);
    }

    // Copy input data to the OpenCL input buffer
    clRawImage->copyPixelsFrom(rawImage);

    scaleRawData(_glsContext, *clRawImage, clScaledRawImage.get(),
                 demosaicParameters->bayerPattern,
                 demosaicParameters->scale_mul,
                 demosaicParameters->black_level / 0xffff);

    NoiseModel* noiseModel = &demosaicParameters->noiseModel;

    LOG_INFO(TAG) << "NoiseLevel: " << demosaicParameters->noiseLevel << std::endl;

    if (calibrateFromImage) {
        noiseModel->rawNlf = MeasureRawNLF(_glsContext, *clScaledRawImage, demosaicParameters->bayerPattern);
    }

    const auto rawVariance = getRawVariance(noiseModel->rawNlf);

    const bool high_noise_image = rawVariance[1][1] > kHighNoiseVariance && !calibrateFromImage;

    std::cout << "Green Channel RAW Variance: " << std::scientific << rawVariance[1][1] << std::endl;

    if (high_noise_image) {
        std::cout << "Despeckeling RAW Image" << std::endl;

        allocateHighNoiseTextures(_glsContext, rawImage.width, rawImage.height);

        bayerToRawRGBA(_glsContext, *clScaledRawImage, rgbaRawImage.get(), demosaicParameters->bayerPattern);

        despeckleRawRGBAImage(_glsContext, *rgbaRawImage, noiseModel->rawNlf.second, denoisedRgbaRawImage.get());

        rawRGBAToBayer(_glsContext, *denoisedRgbaRawImage, clScaledRawImage.get(), demosaicParameters->bayerPattern);
    }

    interpolateGreen(_glsContext, *clScaledRawImage, clGreenImage.get(), demosaicParameters->bayerPattern, rawVariance[1]);

    interpolateRedBlue(_glsContext, *clScaledRawImage, *clGreenImage, clLinearRGBImageA.get(), demosaicParameters->bayerPattern,
                       rawVariance[0], rawVariance[2]);

    // Recover clipped highlights
    blendHighlightsImage(_glsContext, *clLinearRGBImageA, /*clip=*/ 1.0, clLinearRGBImageA.get());

    return clLinearRGBImageA.get();
}

gls::cl_image_2d<gls::rgba_pixel_float>* RawConverter::denoise(DemosaicParameters* demosaicParameters, bool calibrateFromImage) {
    // Convert linear image to YCbCr for denoising
    auto cam_to_ycbcr = cam_ycbcr(demosaicParameters->rgb_cam);

    std::cout << "cam_to_ycbcr: " << std::setprecision(4) << std::scientific << cam_to_ycbcr.span() << std::endl;

    transformImage(_glsContext, *clLinearRGBImageA, clLinearRGBImageA.get(), cam_to_ycbcr);

    NoiseModel* noiseModel = &demosaicParameters->noiseModel;

    // Luma and Chroma Despeckling
    const auto& np = noiseModel->pyramidNlf[0];
    despeckleImage(_glsContext, *clLinearRGBImageA,
                   /*var_a=*/ np.first,
                   /*var_b=*/ np.second,
                   clLinearRGBImageB.get());

    gls::cl_image_2d<gls::rgba_pixel_float>* clDenoisedImage =
        pyramidProcessor->denoise(_glsContext, &(demosaicParameters->denoiseParameters),
                                  clLinearRGBImageB.get(), demosaicParameters->rgb_cam,
                                  &(noiseModel->pyramidNlf), calibrateFromImage);

    if (demosaicParameters->rgbConversionParameters.localToneMapping) {
        const std::array<const gls::cl_image_2d<gls::rgba_pixel_float>*, 3>& guideImage = {
            pyramidProcessor->denoisedImagePyramid[4].get(),
            pyramidProcessor->denoisedImagePyramid[2].get(),
            pyramidProcessor->denoisedImagePyramid[0].get()
        };
        localToneMapping->createMask(_glsContext, *clDenoisedImage, guideImage, *noiseModel, *demosaicParameters);
    }

    // Convert result back to camera RGB
    const auto normalized_ycbcr_to_cam = inverse(cam_to_ycbcr) * demosaicParameters->exposure_multiplier;
    transformImage(_glsContext, *clDenoisedImage, clLinearRGBImageA.get(), normalized_ycbcr_to_cam);

    // High ISO noise texture replacement
    if (clBlueNoise != nullptr) {
        const gls::Vector<2> lumaVariance = { np.first[0], np.second[0] };

        std::cout << "Adding Blue Noise for variance: " << lumaVariance << std::endl;

        const auto grainAmount = 1 + 3 * smoothstep(4e-4, 6e-4, lumaVariance[1]);

        blueNoiseImage(_glsContext, *clLinearRGBImageA, *clBlueNoise, grainAmount * lumaVariance, clLinearRGBImageA.get());
    }

    return clLinearRGBImageA.get();
}

gls::cl_image_2d<gls::rgba_pixel>* RawConverter::postProcess(const DemosaicParameters& demosaicParameters) {
    convertTosRGB(_glsContext, *clLinearRGBImageA, localToneMapping->getMask(), clsRGBImage.get(), demosaicParameters);

    return clsRGBImage.get();
}

gls::cl_image_2d<gls::rgba_pixel>* RawConverter::runPipeline(const gls::image<gls::luma_pixel_16>& rawImage,
                                                             DemosaicParameters* demosaicParameters, bool calibrateFromImage) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // --- Image Demosaicing ---
    demosaic(rawImage, demosaicParameters, calibrateFromImage);

    // --- Image Denoising ---
    denoise(demosaicParameters, calibrateFromImage);

    // --- Image Post Processing ---
    postProcess(*demosaicParameters);

    cl::CommandQueue queue = cl::CommandQueue::getDefault();
    queue.finish();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;

    return clsRGBImage.get();
}

gls::cl_image_2d<gls::rgba_pixel>* RawConverter::runFastPipeline(const gls::image<gls::luma_pixel_16>& rawImage,
                                                                 const DemosaicParameters& demosaicParameters) {
    allocateFastDemosaicTextures(_glsContext, rawImage.width, rawImage.height);

    LOG_INFO(TAG) << "Begin Fast Demosaicing (GPU)..." << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Copy input data to the OpenCL input buffer
    clRawImage->copyPixelsFrom(rawImage);

    // --- Image Demosaicing ---

    scaleRawData(_glsContext, *clRawImage, clScaledRawImage.get(), demosaicParameters.bayerPattern, demosaicParameters.scale_mul,
                 demosaicParameters.black_level / 0xffff);

    fasteDebayer(_glsContext, *clScaledRawImage, clFastLinearRGBImage.get(), demosaicParameters.bayerPattern);

    // Recover clipped highlights
    blendHighlightsImage(_glsContext, *clFastLinearRGBImage, /*clip=*/ 1.0, clFastLinearRGBImage.get());

    // --- Image Post Processing ---

    convertTosRGB(_glsContext, *clFastLinearRGBImage,localToneMapping->getMask(), clsFastRGBImage.get(), demosaicParameters);

    cl::CommandQueue queue = cl::CommandQueue::getDefault();
    queue.finish();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;

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
