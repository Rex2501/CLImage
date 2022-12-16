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

template <size_t levels>
PyramidProcessor<levels>::PyramidProcessor(gls::OpenCLContext* glsContext, int _width, int _height) : width(_width), height(_height), fusedFrames(0) {
    for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
        imagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
        gradientPyramid[i] = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(glsContext->clContext(), width/scale, height/scale);
    }
    for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
        denoisedImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
        subtractedImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
    }
}

gls::Vector<3> nflMultiplier(const DenoiseParameters &denoiseParameters) {
    float luma_mul = denoiseParameters.luma;
    float chroma_mul = denoiseParameters.chroma;
    return { luma_mul, chroma_mul, chroma_mul };
}

#if DEBUG_PYRAMID
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
    out.write_png_file("/Users/fabio/pyramid_7x7" + std::to_string(count++) + ".png");
}

void dumpGradientImage(const gls::cl_image_2d<gls::luma_alpha_pixel_float>& image);

#endif  // DEBUG_PYRAMID

// TODO: Make this a tunable
static const constexpr float lumaDenoiseWeight[4] = {
    1, 1, 1, 1
};

template <size_t levels>
typename PyramidProcessor<levels>::imageType* PyramidProcessor<levels>::denoise(gls::OpenCLContext* glsContext,
                                                                                std::array<DenoiseParameters, levels>* denoiseParameters,
                                                                                const imageType& image,
                                                                                const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                                                                                std::array<YCbCrNLF, levels>* nlfParameters,
                                                                                float exposure_multiplier, bool calibrateFromImage) {
    std::array<gls::Vector<3>, levels> thresholdMultipliers;

    // Create gaussian image pyramid an setup noise model
    for (int i = 0; i < levels; i++) {
        const auto currentLayer = i > 0 ? imagePyramid[i - 1].get() : &image;
        const auto currentGradientLayer = i > 0 ? gradientPyramid[i - 1].get() : &gradientImage;

        if (i < levels - 1) {
            // Generate next layer in the pyramid
            resampleImage(glsContext, "downsampleImageXYZ", *currentLayer, imagePyramid[i].get());
            resampleImage(glsContext, "downsampleImageXY", *currentGradientLayer, gradientPyramid[i].get());
        }

        if (calibrateFromImage) {
            (*nlfParameters)[i] = MeasureYCbCrNLF(glsContext, *currentLayer, exposure_multiplier);
        }

        thresholdMultipliers[i] = nflMultiplier((*denoiseParameters)[i]);
    }

    // Denoise pyramid layers from the bottom to the top, subtracting the noise of the previous layer from the next
    for (int i = levels - 1; i >= 0; i--) {
        const auto denoiseInput = i > 0 ? imagePyramid[i - 1].get() : &image;
        const auto gradientInput = i > 0 ? gradientPyramid[i - 1].get() : &gradientImage;

        if (i < levels - 1) {
            // Subtract the previous layer's noise from the current one
            std::cout << "Reassembling layer " << i + 1 << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;

            const auto np = YCbCrNLF {(*nlfParameters)[i].first * thresholdMultipliers[i], (*nlfParameters)[i].second * thresholdMultipliers[i]};
            subtractNoiseImage(glsContext, *denoiseInput,
                               *(imagePyramid[i]),
                               *(denoisedImagePyramid[i+1]),
                               *gradientInput,
                               lumaDenoiseWeight[i],
                               (*denoiseParameters)[i].sharpening,
                               { np.first[0], np.second[0] },
                               subtractedImagePyramid[i].get());
        }

        std::cout << "Denoising image level " << i << " with multipliers " << thresholdMultipliers[i] << std::endl;

        // Denoise current layer
        denoiseImage(glsContext, i < levels - 1 ? *(subtractedImagePyramid[i]) :  *denoiseInput, *gradientInput,
                     (*nlfParameters)[i].first, (*nlfParameters)[i].second, thresholdMultipliers[i],
                     (*denoiseParameters)[i].chromaBoost, (*denoiseParameters)[i].gradientBoost, (*denoiseParameters)[i].gradientThreshold,
                     denoisedImagePyramid[i].get());
    }

    return denoisedImagePyramid[0].get();
}

// TODO: tunables
const gls::Vector<2> fusionWeights[5] = {
    {1, 4},
    {1, 4},
    {1, 4},
    {1, 4},
    {1, 4}
};

template <size_t levels>
void PyramidProcessor<levels>::fuseFrame(gls::OpenCLContext* glsContext,
                                         std::array<DenoiseParameters, levels>* denoiseParameters,
                                         const imageType& image,
                                         const gls::Matrix<3, 3>& homography,
                                         const gls::cl_image_2d<gls::luma_alpha_pixel_float>& gradientImage,
                                         std::array<YCbCrNLF, levels>* nlfParameters,
                                         float exposure_multiplier, bool calibrateFromImage) {
    std::cout << "Fusing frame " << fusedFrames << std::endl;

    if (fusionImagePyramidA[0] == nullptr) {
        std::cout << "Allocating fusionImagePyramid" << std::endl;
        for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
            fusionImagePyramidA[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
            fusionImagePyramidB[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
            fusionReferenceImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
            fusionReferenceGradientPyramid[i] = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(glsContext->clContext(), width/scale, height/scale);
        }
        fusionBuffer[0] = &fusionImagePyramidA;
        fusionBuffer[1] = &fusionImagePyramidB;
    }

    auto& newFusedImagePyramid = *fusionBuffer[(fusedFrames & 1) == 0];
    auto& previousFusedImagePyramid = *fusionBuffer[(fusedFrames & 1) == 1];

    // Create gaussian image pyramid
    for (int i = 0; i < levels; i++) {
        const auto currentLayer = i > 0 ? (fusedFrames == 0 ? newFusedImagePyramid[i].get() : imagePyramid[i - 1].get()) : &image;
        const auto currentGradientLayer = i > 0 ? gradientPyramid[i - 1].get() : &gradientImage;

        if (fusedFrames == 0 && i == 0) {
            cl::enqueueCopyImage(currentLayer->getImage2D(), newFusedImagePyramid[i]->getImage2D(),
                                 {0, 0, 0}, {0, 0, 0}, {(size_t) currentLayer->width, (size_t) currentLayer->height, 1});

            cl::enqueueCopyImage(currentLayer->getImage2D(), fusionReferenceImagePyramid[i]->getImage2D(),
                                 {0, 0, 0}, {0, 0, 0}, {(size_t) currentLayer->width, (size_t) currentLayer->height, 1});

            cl::enqueueCopyImage(currentGradientLayer->getImage2D(), fusionReferenceGradientPyramid[i]->getImage2D(),
                                 {0, 0, 0}, {0, 0, 0}, {(size_t) currentGradientLayer->width, (size_t) currentGradientLayer->height, 1});
        }

        if (i < levels - 1) {
            // Generate next layer in the pyramid
            resampleImage(glsContext, "downsampleImageXYZ", *currentLayer, fusedFrames == 0 ? newFusedImagePyramid[i+1].get() : imagePyramid[i].get());

            if (fusedFrames == 0) {
                resampleImage(glsContext, "downsampleImageXY", *fusionReferenceGradientPyramid[i], fusionReferenceGradientPyramid[i+1].get());

                cl::enqueueCopyImage(newFusedImagePyramid[i+1]->getImage2D(), fusionReferenceImagePyramid[i+1]->getImage2D(),
                                     {0, 0, 0}, {0, 0, 0}, {(size_t) newFusedImagePyramid[i+1]->width, (size_t) newFusedImagePyramid[i+1]->height, 1});
            }
        }

        if (calibrateFromImage) {
            (*nlfParameters)[i] = MeasureYCbCrNLF(glsContext, *currentLayer, exposure_multiplier);
        }
    }

    if (fusedFrames > 0) {
        for (int i = levels - 1; i >= 0; i--) {
            const auto m = gls::Vector<3> { fusionWeights[i][0], fusionWeights[i][1], fusionWeights[i][1] };
            const auto& np = YCbCrNLF { (*nlfParameters)[i].first * m * m, (*nlfParameters)[i].second * m * m };
            const auto currentLayer = i > 0 ? imagePyramid[i - 1].get() : &image;

            if (i < levels - 1) {
                // Subtract the previous layer's noise from the current one
                std::cout << "Reassembling layer " << i + 1 << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;

                // TODO: The reference image should be the first frame and not be the fused pyramid
                subtractNoiseFusedImage(glsContext, *currentLayer,
                                        *(previousFusedImagePyramid[i+1]),
                                        *(newFusedImagePyramid[i+1]),
                                        subtractedImagePyramid[i].get());
            }

            clFuseFrames(glsContext, *fusionReferenceImagePyramid[i],
                         *fusionReferenceGradientPyramid[i],
                         *subtractedImagePyramid[i],
                         *previousFusedImagePyramid[i],
                         homography,
                         np.first, np.second, fusedFrames, newFusedImagePyramid[i].get());
        }
    }
    fusedFrames++;
}

template <size_t levels>
typename PyramidProcessor<levels>::imageType* PyramidProcessor<levels>::getFusedImage(gls::OpenCLContext* glsContext) {
    auto& newFusedImagePyramid = *fusionBuffer[(fusedFrames & 1) == 1];

    fusedFrames = 0;

    return newFusedImagePyramid[0].get();
}

template struct PyramidProcessor<5>;
