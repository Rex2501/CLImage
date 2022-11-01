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

#ifndef raw_converter_hpp
#define raw_converter_hpp

#include "gls_cl_image.hpp"
#include "pyramidal_denoise.hpp"

class LocalToneMapping {
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr ltmLFAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr ltmMeanLFAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr ltmMFAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr ltmMeanMFAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr ltmHFAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr ltmMeanHFAbGfImage;
    gls::cl_image_2d<gls::luma_pixel_float>::unique_ptr ltmMaskImage;

public:
    LocalToneMapping(gls::OpenCLContext* glsContext) {
        auto clContext = glsContext->clContext();

        // Placeholder, only allocated if LTM is used
        ltmMaskImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, 1, 1);
    }

    void allocateTextures(gls::OpenCLContext* glsContext, int width, int height) {
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

    void createMask(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& image,
                    const std::array<const gls::cl_image_2d<gls::rgba_pixel_float>*, 3>& guideImage,
                    const NoiseModel& noiseModel,
                    const DemosaicParameters& demosaicParameters) {
        const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abImage = {
            ltmLFAbGfImage.get(), ltmMFAbGfImage.get(), ltmHFAbGfImage.get()
        };
        const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abMeanImage = {
            ltmMeanLFAbGfImage.get(), ltmMeanMFAbGfImage.get(), ltmMeanHFAbGfImage.get()
        };

        gls::Vector<2> nlf = { noiseModel.pyramidNlf[0].first[0], noiseModel.pyramidNlf[0].second[0] };
        localToneMappingMask(glsContext, image, guideImage, abImage, abMeanImage, demosaicParameters.ltmParameters, ycbcr_srgb, nlf, ltmMaskImage.get());
    }

    const gls::cl_image_2d<gls::luma_pixel_float>& getMask() {
        return *ltmMaskImage;
    }
};

class RawConverter {
    gls::OpenCLContext* _glsContext;

    // RawConverter base work textures
    gls::cl_image_2d<gls::luma_pixel_16>::unique_ptr clRawImage;
    gls::cl_image_2d<gls::luma_pixel_float>::unique_ptr clScaledRawImage;
    gls::cl_image_2d<gls::luma_pixel_float>::unique_ptr clGreenImage;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clLinearRGBImageA;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clLinearRGBImageB;
    gls::cl_image_2d<gls::rgba_pixel>::unique_ptr clsRGBImage;

    std::unique_ptr<PyramidalDenoise<5>> pyramidalDenoise;

    std::unique_ptr<LocalToneMapping> localToneMapping;

    // RawConverter HighNoise textures
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr rgbaRawImage;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr denoisedRgbaRawImage;
    gls::cl_image_2d<gls::luma_pixel_16>::unique_ptr clBlueNoise;

    // Fast (half resolution) RawConverter textures
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clFastLinearRGBImage;
    gls::cl_image_2d<gls::rgba_pixel>::unique_ptr clsFastRGBImage;

    void allocateTextures(gls::OpenCLContext* glsContext, int width, int height);
    void allocateHighNoiseTextures(gls::OpenCLContext* glsContext, int width, int height);
    void allocateFastDemosaicTextures(gls::OpenCLContext* glsContext, int width, int height);

public:
    RawConverter(gls::OpenCLContext* glsContext) : _glsContext(glsContext) {
        localToneMapping = std::make_unique<LocalToneMapping>(_glsContext);
    }

    gls::cl_image_2d<gls::rgba_pixel>* demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                     DemosaicParameters* demosaicParameters, bool calibrateFromImage = false);

    gls::cl_image_2d<gls::rgba_pixel>* fastDemosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                         const DemosaicParameters& demosaicParameters);

    static gls::image<gls::rgb_pixel>::unique_ptr convertToRGBImage(const gls::cl_image_2d<gls::rgba_pixel>& clRGBAImage);

};

#endif /* raw_converter_hpp */
