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
#include "pyramid_processor.hpp"

class LocalToneMapping {
    gls::cl_image_2d<gls::luma_pixel_float>::unique_ptr ltmMaskImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr lfAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr lfAbGfMeanImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr mfAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr mfAbGfMeanImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr hfAbGfImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr hfAbGfMeanImage;

public:
    LocalToneMapping(gls::OpenCLContext* glsContext) {
        auto clContext = glsContext->clContext();

        // Placeholder, only allocated if LTM is used
        ltmMaskImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, 1, 1);
    }

    void allocateTextures(gls::OpenCLContext* glsContext, int width, int height) {
        auto clContext = glsContext->clContext();

        if (ltmMaskImage->width != width || ltmMaskImage->height != height) {
            ltmMaskImage    = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
            lfAbGfImage     = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/16, height/16);
            lfAbGfMeanImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/16, height/16);
            mfAbGfImage     = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/4, height/4);
            mfAbGfMeanImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width/4, height/4);
            hfAbGfImage     = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width, height);
            hfAbGfMeanImage = std::make_unique<gls::cl_image_2d<gls::luma_alpha_pixel_float>>(clContext, width, height);
        }
    }

    void createMask(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& image,
                    const std::array<const gls::cl_image_2d<gls::rgba_pixel_float>*, 3>& guideImage,
                    const NoiseModel<5>& noiseModel,
                    const DemosaicParameters& demosaicParameters) {
        const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abImage = {
            lfAbGfImage.get(), mfAbGfImage.get(), hfAbGfImage.get()
        };
        const std::array<const gls::cl_image_2d<gls::luma_alpha_pixel_float>*, 3>& abMeanImage = {
            lfAbGfMeanImage.get(), mfAbGfMeanImage.get(), hfAbGfMeanImage.get()
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

    // TODO: this should probably be camera specific
    static const constexpr float kHighNoiseVariance = 2.5e-04;

    // RawConverter base work textures
    gls::cl_image_2d<gls::luma_pixel_16>::unique_ptr clRawImage;
    gls::cl_image_2d<gls::luma_pixel_float>::unique_ptr clScaledRawImage;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clRawSobelImage;
    gls::cl_image_2d<gls::luma_alpha_pixel_float>::unique_ptr clRawGradientImage;
    gls::cl_image_2d<gls::luma_pixel_float>::unique_ptr clGreenImage;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clLinearRGBImageA;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clLinearRGBImageB;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clsRGBImage;

    std::unique_ptr<PyramidProcessor<5>> pyramidProcessor;

    std::unique_ptr<LocalToneMapping> localToneMapping;

    // RawConverter HighNoise textures
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr rgbaRawImage;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr denoisedRgbaRawImage;
    gls::cl_image_2d<gls::luma_pixel_16>::unique_ptr clBlueNoise;

    // Fast (half resolution) RawConverter textures
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clFastLinearRGBImage;
    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr clsFastRGBImage;

    void allocateTextures(gls::OpenCLContext* glsContext, int width, int height);
    void allocateHighNoiseTextures(gls::OpenCLContext* glsContext, int width, int height);
    void allocateFastDemosaicTextures(gls::OpenCLContext* glsContext, int width, int height);

public:
    RawConverter(gls::OpenCLContext* glsContext) : _glsContext(glsContext) {
        localToneMapping = std::make_unique<LocalToneMapping>(_glsContext);
    }

    gls::OpenCLContext* getContext() const {
        return _glsContext;
    }

    gls::cl_image_2d<gls::rgba_pixel_float>* runPipeline(const gls::image<gls::luma_pixel_16>& rawImage,
                                                         DemosaicParameters* demosaicParameters, bool calibrateFromImage = false);

    gls::cl_image_2d<gls::rgba_pixel_float>* demosaic(const gls::image<gls::luma_pixel_16>& rawImage,
                                                      DemosaicParameters* demosaicParameters, bool calibrateFromImage);

    gls::cl_image_2d<gls::rgba_pixel_float>* denoise(const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                                                     DemosaicParameters* demosaicParameters, bool calibrateFromImage);

    void fuseFrame(const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage, const gls::Matrix<3, 3>& homography,
                   DemosaicParameters* demosaicParameters, bool calibrateFromImage);

    gls::cl_image_2d<gls::rgba_pixel_float>* getFusedImage();

    gls::cl_image_2d<gls::rgba_pixel_float>* postProcess(const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                                                         const DemosaicParameters& demosaicParameters);

    gls::cl_image_2d<gls::rgba_pixel_float>* runFastPipeline(const gls::image<gls::luma_pixel_16>& rawImage,
                                                             const DemosaicParameters& demosaicParameters);

    template <typename T = gls::rgb_pixel>
    static typename gls::image<T>::unique_ptr convertToRGBImage(const gls::cl_image_2d<gls::rgba_pixel_float>& clRGBAImage);
};

#endif /* raw_converter_hpp */
