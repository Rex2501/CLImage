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

#include "pyramid_processor.hpp"

#include <iomanip>

template
PyramidProcessor<5>::PyramidProcessor(gls::OpenCLContext* glsContext, int width, int height);

struct BilateralDenoiser : ImageDenoiser {
    BilateralDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                 const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                 const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                 float chromaBoost, float gradientBoost, int pyramidLevel,
              gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImage(glsContext, inputImage, var_a, var_b, chromaBoost, gradientBoost, outputImage);
    }
};

struct GuidedFastDenoiser : ImageDenoiser {
    GuidedFastDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                 const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                 const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                 float chromaBoost, float gradientBoost, int pyramidLevel,
                 gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImageGuided(glsContext, inputImage, var_a, var_b, outputImage);
    }
};

template <size_t levels>
PyramidProcessor<levels>::PyramidProcessor(gls::OpenCLContext* glsContext, int _width, int _height) : width(_width), height(_height), fusedFrames(0) {
    for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
        imagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
    }
    for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
        denoisedImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
        denoiser[i] = std::make_unique<BilateralDenoiser>(glsContext, width/scale, height/scale);
    }
}

gls::Vector<3> nflMultiplier(const DenoiseParameters &denoiseParameters) {
    float luma_mul = denoiseParameters.luma * denoiseParameters.luma;
    float chroma_mul = denoiseParameters.chroma * denoiseParameters.chroma;
    return { luma_mul, chroma_mul, chroma_mul };
}

extern const gls::Matrix<3, 3> ycbcr_srgb;

void dumpYCbCrImage(const gls::cl_image_2d<gls::rgba_pixel_float>& image) {
    gls::image<gls::rgb_pixel> out(image.width, image.height);
    const auto downsampledCPU = image.mapImage();
    out.apply([&downsampledCPU](gls::rgb_pixel* p, int x, int y){
        const auto& ip = downsampledCPU[y][x];
        const auto& v = ycbcr_srgb * gls::Vector<3>{ip.x, ip.y, ip.z};
        *p = gls::rgb_pixel {
            (uint8_t) (255 * std::sqrt(std::clamp(v[0], 0.0f, 1.0f))),
            (uint8_t) (255 * std::sqrt(std::clamp(v[1], 0.0f, 1.0f))),
            (uint8_t) (255 * std::sqrt(std::clamp(v[2], 0.0f, 1.0f)))
        };
    });
    image.unmapImage(downsampledCPU);
    static int count = 1;
    out.write_png_file("/Users/fabio/pyramid" + std::to_string(count++) + ".png");
}

template <size_t levels>
typename PyramidProcessor<levels>::imageType* PyramidProcessor<levels>::denoise(gls::OpenCLContext* glsContext,
                                                                                std::array<DenoiseParameters, levels>* denoiseParameters,
                                                                                imageType* image, std::array<YCbCrNLF, levels>* nlfParameters,
                                                                                float exposure_multiplier, bool calibrateFromImage) {
    std::array<YCbCrNLF, levels> calibrated_nlf;

    // Create gaussian image pyramid an setup noise model
    for (int i = 0; i < levels; i++) {
        const auto currentLayer = i > 0 ? imagePyramid[i - 1].get() : image;

        if (i < levels - 1) {
            // Generate next layer in the pyramid
            resampleImage(glsContext, "downsampleImage", *currentLayer, imagePyramid[i].get());
        }

        if (calibrateFromImage) {
            const auto nlf = MeasureYCbCrNLF(glsContext, *currentLayer, exposure_multiplier);
            (*nlfParameters)[i] = nlf;
        }

        const auto m = nflMultiplier((*denoiseParameters)[i]);

        calibrated_nlf[i] = { (*nlfParameters)[i].first * m, (*nlfParameters)[i].second * m };
    }

    // Denoise all pyramid layers independently
    for (int i = 0; i < levels; i++) {
        const auto denoiseInput = i > 0 ? imagePyramid[i - 1].get() : image;

        // Denoise current layer
        const auto& np = calibrated_nlf[i];
        denoiser[i]->denoise(glsContext, *denoiseInput,
                             np.first, np.second,
                             (*denoiseParameters)[i].chromaBoost, (*denoiseParameters)[i].gradientBoost, /*pyramidLevel=*/ i,
                             denoisedImagePyramid[i].get());
    }

    // Reassemble pyramyd from the bottom up
    for (int i = levels - 2; i >= 0; i--) {
        const auto& np = calibrated_nlf[i];

        // Subtract the previous layer's noise from the current one
        std::cout << "Reassembling layer " << i << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;
        reassembleImage(glsContext, *(denoisedImagePyramid[i]),
                        *(imagePyramid[i]),
                        *(denoisedImagePyramid[i+1]),
                        (*denoiseParameters)[i].sharpening, { np.first[0], np.second[0] },
                        denoisedImagePyramid[i].get());
    }

    return denoisedImagePyramid[0].get();
}

template
typename PyramidProcessor<5>::imageType* PyramidProcessor<5>::denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, 5>* denoiseParameters,
                                                                      imageType* image, std::array<YCbCrNLF, 5>* nlfParameters,
                                                                      float exposure_multiplier, bool calibrateFromImage);
const float fusionWeights[5] = { 2, 1, 0.5, 0.25, 0.125 };

template <size_t levels>
void PyramidProcessor<levels>::fuseFrame(gls::OpenCLContext* glsContext,
                                         std::array<DenoiseParameters, levels>* denoiseParameters,
                                         const imageType& image, std::array<YCbCrNLF, levels>* nlfParameters,
                                         float exposure_multiplier, bool calibrateFromImage) {
    std::cout << "Fusing frame " << fusedFrames << std::endl;

    if (fusionImagePyramid[0] == nullptr) {
        std::cout << "Allocating fusionImagePyramid" << std::endl;
        for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
            fusionImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
        }
    }

    auto& newFusedImagePyramid = (fusedFrames & 1) == 0 ? fusionImagePyramid : denoisedImagePyramid;
    auto& previousFusedImagePyramid = (fusedFrames & 1) == 1 ? fusionImagePyramid : denoisedImagePyramid;

    std::array<YCbCrNLF, levels> calibrated_nlf;

    // Create gaussian image pyramid an setup noise model
    for (int i = 0; i < levels; i++) {
        const auto currentLayer = i > 0 ? imagePyramid[i - 1].get() : &image;

        if (i < levels - 1) {
            // Generate next layer in the pyramid
            resampleImage(glsContext, "downsampleImage", *currentLayer, imagePyramid[i].get());
        }
        if (fusedFrames == 0) {
            cl::enqueueCopyImage(currentLayer->getImage2D(), newFusedImagePyramid[i]->getImage2D(),
                                 {0, 0, 0}, {0, 0, 0}, {(size_t) currentLayer->width, (size_t) currentLayer->height, 1});
        }

        if (calibrateFromImage) {
            const auto nlf = MeasureYCbCrNLF(glsContext, *currentLayer, exposure_multiplier);
            (*nlfParameters)[i] = nlf;
        }

        const auto m = gls::Vector<3> {1, 4.0f, 4.0f};
        calibrated_nlf[i] = { (*nlfParameters)[i].first * m, (*nlfParameters)[i].second * m };
    }

    if (fusedFrames > 0) {
        for (int i = levels - 1; i >= 0; i--) {
            const auto& np = calibrated_nlf[i];
            const auto currentLayer = i > 0 ? imagePyramid[i - 1].get() : &image;
            clFuseFrames(glsContext, *currentLayer, *previousFusedImagePyramid[i],
                         np.first, np.second, fusedFrames, newFusedImagePyramid[i].get());

        }

        // Reassemble pyramyd from the bottom up
        for (int i = levels - 2; i >= 0; i--) {
            // Subtract the previous layer's noise from the current one
            std::cout << "Reassembling Fused layer " << i << std::endl;
            reassembleFusedImage(glsContext, *(newFusedImagePyramid[i]),
                                 *(previousFusedImagePyramid[i+1]),
                                 *(newFusedImagePyramid[i+1]),
                                 newFusedImagePyramid[i].get());
        }
    }
    fusedFrames++;
}

template
void PyramidProcessor<5>::fuseFrame(gls::OpenCLContext* glsContext,
                                    std::array<DenoiseParameters, 5>* denoiseParameters,
                                    const imageType& image, std::array<YCbCrNLF, 5>* nlfParameters,
                                    float exposure_multiplier, bool calibrateFromImage);

template <size_t levels>
typename PyramidProcessor<levels>::imageType* PyramidProcessor<levels>::getFusedImage(gls::OpenCLContext* glsContext) {
    auto& newFusedImagePyramid = (fusedFrames & 1) == 1 ? fusionImagePyramid : denoisedImagePyramid;

    std::cout << "Returning fused image" << std::endl;

    fusedFrames = 0;

    return newFusedImagePyramid[0].get();
}

template
typename PyramidProcessor<5>::imageType* PyramidProcessor<5>::getFusedImage(gls::OpenCLContext* glsContext);
