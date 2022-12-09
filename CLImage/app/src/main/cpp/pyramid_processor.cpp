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
    out.write_png_file("/Users/fabio/pyramid" + std::to_string(count++) + ".png");
}

void dumpGradientImage(const gls::cl_image_2d<gls::luma_alpha_pixel_float>& image, float nlfa, float nlfb) {
    gls::image<gls::luma_pixel> out(image.width, image.height);
    const auto image_cpu = image.mapImage();
    out.apply([&](gls::luma_pixel* p, int x, int y) {
        const auto& ip = image_cpu[y][x];

        const float sigma = sqrt(nlfa + ip.x * nlfb);

        *p = gls::luma_pixel {
            (uint8_t) (255 * std::sqrt(std::clamp((float) (ip.w > 4 * sigma ? ip.w : 0), 0.0f, 1.0f)))
        };
    });
    image.unmapImage(image_cpu);
    static int count = 1;
    out.write_png_file("/Users/fabio/gradient_lfb" + std::to_string(count++) + ".png");
}
#endif  // DEBUG_PYRAMID

void dumpGradientImage(const gls::cl_image_2d<gls::luma_alpha_pixel_float>& image);

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

    // Denoise all pyramid layers independently
    for (int i = levels - 1; i >= 0; i--) {
        const auto denoiseInput = i > 0 ? imagePyramid[i - 1].get() : &image;
        const auto gradientInput = i > 0 ? gradientPyramid[i - 1].get() : &gradientImage;

        std::cout << "Denoising image level " << i << " with multipliers " << thresholdMultipliers[i] << std::endl;

        // dumpGradientImage(*gradientInput);

        // Denoise current layer
        denoiseImage(glsContext, *denoiseInput, *gradientInput,
                     (*nlfParameters)[i].first, (*nlfParameters)[i].second, thresholdMultipliers[i],
                     (*denoiseParameters)[i].chromaBoost, (*denoiseParameters)[i].gradientBoost, i,
                     denoisedImagePyramid[i].get());
    }

    // Reassemble pyramyd from the bottom up
    for (int i = levels - 2; i >= 0; i--) {
        // TODO: this really is a kludge
        const auto np = YCbCrNLF {(*nlfParameters)[i].first * thresholdMultipliers[i], (*nlfParameters)[i].second * thresholdMultipliers[i]};

        // Subtract the previous layer's noise from the current one
        std::cout << "Reassembling layer " << i << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;
        reassembleImage(glsContext, *(denoisedImagePyramid[i]),
                        *(imagePyramid[i]),
                        *(denoisedImagePyramid[i+1]),
                        (*denoiseParameters)[i].sharpening, { np.first[0], np.second[0] },
                        denoisedImagePyramid[i].get());
    }

    const float nlfa = (*nlfParameters)[0].first[0];
    const float nlfb = (*nlfParameters)[0].second[0];

    // dumpGradientImage(*denoisedImagePyramid[0], nlfa, nlfb);

    gls::cl_image_2d<gls::rgba_pixel_float> blurredImage(glsContext->clContext(), image.size());
    gaussianBlurImage(glsContext, image, 1, &blurredImage);

    const auto image_cpu = image.mapImage();
    const auto blurredImage_cpu = blurredImage.mapImage();
    denoisedImagePyramid[0]->apply([&](gls::rgba_pixel_float *p, int x, int y){
        const float sigma = sqrt(nlfa + p->x * nlfb);
        const float high_gradient = smoothstep(sigma, 4 * sigma, p->w);

        p->x += 0.5 * high_gradient * (image_cpu[y][x][0] - blurredImage_cpu[y][x][0]);
        // p->x = std::lerp(p->x, image_cpu[y][x][0], high_gradient);
        // p->red = image_cpu[y][x][0];
    });
    image.unmapImage(image_cpu);
    blurredImage.unmapImage(blurredImage_cpu);

    return denoisedImagePyramid[0].get();
}

// TODO: tunables
const gls::Vector<2> fusionWeights[5] = {
    {4, 4},
    {4, 4},
    {4, 4},
    {4, 4},
    {4, 4}
};

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
        fusionBuffer[0] = &denoisedImagePyramid;
        fusionBuffer[1] = &fusionImagePyramid;
    }

    auto& newFusedImagePyramid = *fusionBuffer[(fusedFrames & 1) == 0];
    auto& previousFusedImagePyramid = *fusionBuffer[(fusedFrames & 1) == 1];

    // Create gaussian image pyramid
    for (int i = 0; i < levels; i++) {
        const auto currentLayer = i > 0 ? (fusedFrames == 0 ? newFusedImagePyramid[i].get() : imagePyramid[i - 1].get()) : &image;

        if (i < levels - 1) {
            // Generate next layer in the pyramid
            resampleImage(glsContext, "downsampleImageXYZ", *currentLayer, fusedFrames == 0 ? newFusedImagePyramid[i+1].get() : imagePyramid[i].get());
        }
        if (i == 0 && fusedFrames == 0) {
            cl::enqueueCopyImage(currentLayer->getImage2D(), newFusedImagePyramid[i]->getImage2D(),
                                 {0, 0, 0}, {0, 0, 0}, {(size_t) currentLayer->width, (size_t) currentLayer->height, 1});
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

template <size_t levels>
typename PyramidProcessor<levels>::imageType* PyramidProcessor<levels>::getFusedImage(gls::OpenCLContext* glsContext) {
    auto& newFusedImagePyramid = *fusionBuffer[(fusedFrames & 1) == 1];

    fusedFrames = 0;

    return newFusedImagePyramid[0].get();
}

template struct PyramidProcessor<5>;
