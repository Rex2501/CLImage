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

#include "CameraCalibration.hpp"

#include <array>
#include <cmath>
#include <filesystem>

#include "demosaic.hpp"

template <size_t levels = 5>
class Sonya6400Calibration : public CameraCalibration<levels> {
    static const std::array<NoiseModel<levels>, 11> NLFData;

public:
    NoiseModel<levels> nlfFromIso(int iso) const override {
        iso = std::clamp(iso, 100, 102400);
        if (iso >= 100 && iso < 200) {
            float a = (iso - 100) / 100;
            return lerp<levels>(NLFData[0], NLFData[1], a);
        } else if (iso >= 200 && iso < 400) {
            float a = (iso - 200) / 200;
            return lerp<levels>(NLFData[1], NLFData[2], a);
        } else if (iso >= 400 && iso < 800) {
            float a = (iso - 400) / 400;
            return lerp<levels>(NLFData[2], NLFData[3], a);
        } else if (iso >= 800 && iso < 1600) {
            float a = (iso - 800) / 800;
            return lerp<levels>(NLFData[3], NLFData[4], a);
        } else if (iso >= 1600 && iso < 3200) {
            float a = (iso - 1600) / 1600;
            return lerp<levels>(NLFData[4], NLFData[5], a);
        } else if (iso >= 3200 && iso < 6400) {
            float a = (iso - 3200) / 3200;
            return lerp<levels>(NLFData[5], NLFData[6], a);
        } else if (iso >= 6400 && iso < 12800) {
            float a = (iso - 6400) / 6400;
            return lerp<levels>(NLFData[6], NLFData[7], a);
        } else if (iso >= 12800 && iso < 25600) {
            float a = (iso - 12800) / 12800;
            return lerp<levels>(NLFData[7], NLFData[8], a);
        } else if (iso >= 25600 && iso < 51200) {
            float a = (iso - 25600) / 25600;
            return lerp<levels>(NLFData[8], NLFData[9], a);
        } else /* if (iso >= 51200 && iso <= 102400) */ {
            float a = (iso - 51200) / 51200;
            return lerp<levels>(NLFData[9], NLFData[10], a);
        }
    }

    std::pair<float, std::array<DenoiseParameters, levels>> getDenoiseParameters(int iso) const override {
        const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(102400) - log2(100)), 0.0, 1.0);

        std::cout << "Sonya6400 DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

        float lerp = std::lerp(0.125f, 2.0f, nlf_alpha);
        float lerp_c = 1;

        float lmult[5] = { 0.5, 1, 0.5, 0.25, 0.125 };
        float cmult[5] = { 1, 0.5, 0.5, 0.5, 0.25 };

        float chromaBoost = 8;

        float gradientBoost = smoothstep(0.3, 0.8, nlf_alpha);
        float gradientThreshold = 1 + 0.5 * smoothstep(0.3, 0.8, nlf_alpha);

        std::array<DenoiseParameters, 5> denoiseParameters = {{
            {
                .luma = lmult[0] * lerp,
                .chroma = cmult[0] * lerp_c,
                .chromaBoost = chromaBoost,
                .gradientBoost = 8 * gradientBoost,
                .gradientThreshold = gradientThreshold,
                .sharpening = std::lerp(1.5f, 1.0f, nlf_alpha)
            },
            {
                .luma = lmult[1] * lerp,
                .chroma = cmult[1] * lerp_c,
                .chromaBoost = chromaBoost,
                .gradientBoost = gradientBoost,
                .sharpening = 1.1
            },
            {
                .luma = lmult[2] * lerp,
                .chroma = cmult[2] * lerp_c,
                .chromaBoost = chromaBoost,
                .gradientBoost = gradientBoost,
                .sharpening = 1
            },
            {
                .luma = lmult[3] * lerp,
                .chroma = cmult[3] * lerp_c,
                .chromaBoost = chromaBoost,
                .gradientBoost = gradientBoost,
                .sharpening = 1
            },
            {
                .luma = lmult[4] * lerp,
                .chroma = cmult[4] * lerp_c,
                .chromaBoost = chromaBoost,
                .gradientBoost = gradientBoost,
                .sharpening = 1
            }
        }};

        return { nlf_alpha, denoiseParameters };
    }


    DemosaicParameters buildDemosaicParameters() const override {
        return {
            .rgbConversionParameters = {
                // .exposureBias = -1.0,
                // .blacks = 0.1,
                .contrast = 1.05,
                .saturation = 1.0,
                .toneCurveSlope = 3.5,
                .localToneMapping = false
            },
            .ltmParameters = {
                .eps = 0.01,
                .shadows = 1.0, // 0.8,
                .highlights = 1.0, // 1.5,
                .detail = { 1, 1.2, 2.0 }
            }
        };
    }

    void calibrate(RawConverter* rawConverter, const std::filesystem::path& input_dir) const override {
        std::array<CalibrationEntry, 11> calibration_files = {{
            { 100,    "DSC00185_ISO_100.DNG",    { 2435, 521, 1109, 732 }, false },
            { 200,    "DSC00188_ISO_200.DNG",    { 2435, 521, 1109, 732 }, false },
            { 400,    "DSC00192_ISO_400.DNG",    { 2435, 521, 1109, 732 }, false },
            { 800,    "DSC00195_ISO_800.DNG",    { 2435, 521, 1109, 732 }, false },
            { 1600,   "DSC00198_ISO_1600.DNG",   { 2435, 521, 1109, 732 }, false },
            { 3200,   "DSC00201_ISO_3200.DNG",   { 2435, 521, 1109, 732 }, false },
            { 6400,   "DSC00204_ISO_6400.DNG",   { 2435, 521, 1109, 732 }, false },
            { 12800,  "DSC00207_ISO_12800.DNG",  { 2435, 521, 1109, 732 }, false },
            { 25600,  "DSC00210_ISO_25600.DNG",  { 2435, 521, 1109, 732 }, false },
            { 51200,  "DSC00227_ISO_51200.DNG",  { 2435, 521, 1109, 732 }, false },
            { 102400, "DSC00230_ISO_102400.DNG", { 2435, 521, 1109, 732 }, false },
        }};

        std::array<NoiseModel<5>, 11> noiseModel;

        for (int i = 0; i < calibration_files.size(); i++) {
            auto& entry = calibration_files[i];
            const auto input_path = input_dir / entry.fileName;

            DemosaicParameters demosaicParameters = {
                .rgbConversionParameters = {
                    .localToneMapping = false
                }
            };

            const auto rgb_image = CameraCalibration<5>::calibrate(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
            rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal.png", /*skip_alpha=*/ true);

            noiseModel[i] = demosaicParameters.noiseModel;
        }

        std::cout << "// iPhone 11 Calibration table:" << std::endl;
        dumpNoiseModel(calibration_files, noiseModel);
    }
};

void calibrateSonya6400(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    Sonya6400Calibration calibration;
    calibration.calibrate(rawConverter, input_dir);
}

template <typename T>
typename gls::image<T>::unique_ptr demosaicSonya6400RawImage(RawConverter* rawConverter,
                                                             gls::tiff_metadata* dng_metadata,
                                                             gls::tiff_metadata* exif_metadata,
                                                             const gls::image<gls::luma_pixel_16>& inputImage) {
    Sonya6400Calibration calibration;
    auto demosaicParameters = calibration.getDemosaicParameters(inputImage, dng_metadata, exif_metadata);

    unpackDNGMetadata(inputImage, dng_metadata, demosaicParameters.get(), /*auto_white_balance=*/ false, nullptr, false);

    const auto demosaicedImage = rawConverter->runPipeline(inputImage, demosaicParameters.get(), /*calibrateFromImage=*/ false);

//    gls::cl_image_2d<gls::rgba_pixel_float> unsquishedImage(rawConverter->getContext()->clContext(), demosaicedImage->width, demosaicedImage->height * 1.2);
//    clRescaleImage(rawConverter->getContext(), *demosaicedImage, &unsquishedImage);

    return RawConverter::convertToRGBImage<T>(*demosaicedImage);
}

template
typename gls::image<gls::rgb_pixel>::unique_ptr demosaicSonya6400RawImage<gls::rgb_pixel>(RawConverter* rawConverter,
                                                                                          gls::tiff_metadata* dng_metadata,
                                                                                          gls::tiff_metadata* exif_metadata,
                                                                                          const gls::image<gls::luma_pixel_16>& inputImage);

template
typename gls::image<gls::rgb_pixel_16>::unique_ptr demosaicSonya6400RawImage<gls::rgb_pixel_16>(RawConverter* rawConverter,
                                                                                                gls::tiff_metadata* dng_metadata,
                                                                                                gls::tiff_metadata* exif_metadata,
                                                                                                const gls::image<gls::luma_pixel_16>& inputImage);

gls::image<gls::rgb_pixel>::unique_ptr demosaicSonya6400DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    return demosaicSonya6400RawImage(rawConverter, &dng_metadata, &exif_metadata, *inputImage);
}

// --- NLFData ---

template<>
const std::array<NoiseModel<5>, 11> Sonya6400Calibration<5>::NLFData = {{
    // ISO 100
    {
        {{1.000e-08, 1.000e-08, 1.000e-08, 1.000e-08}, {3.681e-04, 3.448e-04, 2.846e-04, 3.447e-04}},
        {{
            {{1.000e-08, 1.000e-08, 1.026e-08}, {3.686e-04, 7.427e-06, 8.743e-06}},
            {{1.000e-08, 1.000e-08, 1.000e-08}, {5.010e-04, 8.027e-06, 7.757e-06}},
            {{1.000e-08, 7.791e-08, 6.431e-08}, {7.489e-04, 6.842e-06, 6.428e-06}},
            {{1.000e-08, 2.936e-07, 3.165e-07}, {9.712e-04, 4.746e-06, 3.824e-06}},
            {{1.000e-08, 7.608e-07, 7.625e-07}, {1.510e-03, 7.559e-06, 5.740e-06}},
        }}
    },
    // ISO 200
    {
        {{1.000e-08, 1.000e-08, 1.000e-08, 1.000e-08}, {4.499e-04, 3.647e-04, 3.140e-04, 3.643e-04}},
        {{
            {{1.000e-08, 1.000e-08, 3.841e-08}, {3.891e-04, 9.280e-06, 1.321e-05}},
            {{1.000e-08, 1.000e-08, 2.128e-08}, {5.035e-04, 9.215e-06, 1.053e-05}},
            {{1.000e-08, 7.166e-08, 6.374e-08}, {7.476e-04, 7.374e-06, 7.463e-06}},
            {{1.000e-08, 2.895e-07, 3.087e-07}, {9.677e-04, 4.947e-06, 4.188e-06}},
            {{1.000e-08, 7.605e-07, 7.614e-07}, {1.500e-03, 7.653e-06, 5.846e-06}},
        }}
    },
    // ISO 400
    {
        {{1.000e-08, 1.000e-08, 1.000e-08, 1.000e-08}, {6.396e-04, 4.238e-04, 3.977e-04, 4.217e-04}},
        {{
            {{1.000e-08, 1.000e-08, 7.723e-08}, {4.263e-04, 1.306e-05, 2.182e-05}},
            {{1.000e-08, 1.000e-08, 1.239e-08}, {5.113e-04, 1.186e-05, 1.652e-05}},
            {{1.000e-08, 6.741e-08, 5.787e-08}, {7.483e-04, 8.636e-06, 9.625e-06}},
            {{1.000e-08, 2.841e-07, 2.978e-07}, {9.675e-04, 5.337e-06, 4.905e-06}},
            {{1.000e-08, 7.600e-07, 7.671e-07}, {1.494e-03, 7.795e-06, 5.757e-06}},
        }}
    },
    // ISO 800
    {
        {{1.000e-08, 1.000e-08, 1.000e-08, 1.000e-08}, {1.018e-03, 5.322e-04, 5.686e-04, 5.297e-04}},
        {{
            {{1.000e-08, 6.604e-08, 2.035e-07}, {4.924e-04, 2.036e-05, 3.801e-05}},
            {{1.000e-08, 1.000e-08, 3.925e-08}, {5.207e-04, 1.685e-05, 2.784e-05}},
            {{1.000e-08, 7.534e-08, 5.754e-08}, {7.531e-04, 1.069e-05, 1.390e-05}},
            {{1.000e-08, 2.722e-07, 2.967e-07}, {9.571e-04, 6.101e-06, 5.980e-06}},
            {{1.000e-08, 7.695e-07, 7.782e-07}, {1.472e-03, 8.027e-06, 6.125e-06}},
        }}
    },
    // ISO 1600
    {
        {{1.000e-08, 1.000e-08, 1.000e-08, 1.000e-08}, {1.717e-03, 7.571e-04, 9.241e-04, 7.550e-04}},
        {{
            {{1.000e-08, 1.857e-07, 3.453e-07}, {6.393e-04, 3.485e-05, 7.274e-05}},
            {{1.000e-08, 8.949e-08, 1.677e-07}, {5.405e-04, 2.620e-05, 4.922e-05}},
            {{1.000e-08, 8.995e-08, 7.926e-08}, {7.525e-04, 1.464e-05, 2.207e-05}},
            {{1.000e-08, 2.768e-07, 3.090e-07}, {9.602e-04, 7.266e-06, 8.327e-06}},
            {{1.000e-08, 7.649e-07, 7.682e-07}, {1.473e-03, 8.244e-06, 6.902e-06}},
        }}
    },
    // ISO 3200
    {
        {{1.660e-05, 1.000e-08, 1.000e-08, 1.000e-08}, {2.888e-03, 1.320e-03, 1.644e-03, 1.325e-03}},
        {{
            {{1.000e-08, 5.176e-07, 7.774e-07}, {9.818e-04, 6.376e-05, 1.439e-04}},
            {{1.000e-08, 2.866e-07, 4.436e-07}, {5.614e-04, 4.559e-05, 9.371e-05}},
            {{1.000e-08, 1.398e-07, 1.546e-07}, {7.514e-04, 2.278e-05, 3.856e-05}},
            {{1.000e-08, 2.900e-07, 3.274e-07}, {9.181e-04, 9.926e-06, 1.301e-05}},
            {{1.000e-08, 8.220e-07, 8.594e-07}, {1.381e-03, 8.419e-06, 7.551e-06}},
        }}
    },
    // ISO 6400
    {
        {{4.137e-05, 1.000e-08, 4.311e-06, 1.000e-08}, {5.539e-03, 2.345e-03, 3.110e-03, 2.352e-03}},
        {{
            {{1.634e-06, 1.265e-06, 1.432e-06}, {1.585e-03, 1.157e-04, 2.866e-04}},
            {{1.000e-08, 6.595e-07, 9.218e-07}, {6.463e-04, 8.421e-05, 1.826e-04}},
            {{1.000e-08, 2.738e-07, 3.553e-07}, {7.587e-04, 3.806e-05, 6.996e-05}},
            {{1.000e-08, 3.175e-07, 3.725e-07}, {8.731e-04, 1.458e-05, 2.216e-05}},
            {{1.000e-08, 9.064e-07, 9.689e-07}, {1.236e-03, 8.252e-06, 8.845e-06}},
        }}
    },
    // ISO 12800
    {
        {{6.766e-05, 2.051e-06, 1.645e-05, 2.807e-06}, {1.142e-02, 4.548e-03, 6.092e-03, 4.585e-03}},
        {{
            {{2.206e-05, 3.045e-06, 3.033e-06}, {2.528e-03, 2.095e-04, 5.749e-04}},
            {{1.000e-08, 1.672e-06, 1.998e-06}, {8.673e-04, 1.579e-04, 3.680e-04}},
            {{1.000e-08, 5.859e-07, 7.594e-07}, {7.845e-04, 6.907e-05, 1.366e-04}},
            {{1.000e-08, 3.830e-07, 4.875e-07}, {9.025e-04, 2.413e-05, 4.117e-05}},
            {{1.000e-08, 8.809e-07, 9.512e-07}, {1.303e-03, 1.116e-05, 1.463e-05}},
        }}
    },
    // ISO 25600
    {
        {{1.579e-04, 1.000e-08, 1.000e-08, 3.245e-07}, {1.696e-02, 9.136e-03, 1.286e-02, 9.248e-03}},
        {{
            {{7.544e-05, 7.086e-06, 3.769e-06}, {4.212e-03, 3.907e-04, 1.213e-03}},
            {{1.000e-08, 3.283e-06, 3.365e-06}, {1.327e-03, 3.095e-04, 7.497e-04}},
            {{1.000e-08, 1.072e-06, 1.201e-06}, {8.396e-04, 1.341e-04, 2.781e-04}},
            {{1.000e-08, 5.847e-07, 7.495e-07}, {8.875e-04, 4.236e-05, 7.692e-05}},
            {{1.000e-08, 9.475e-07, 1.120e-06}, {1.221e-03, 1.611e-05, 2.217e-05}},
        }}
    },
    // ISO 51200
    {
        {{2.426e-04, 1.000e-08, 1.693e-06, 1.000e-08}, {1.800e-02, 1.339e-02, 1.708e-02, 1.357e-02}},
        {{
            {{1.611e-04, 1.476e-05, 1.097e-05}, {4.615e-03, 5.094e-04, 1.937e-03}},
            {{7.772e-06, 6.942e-06, 8.358e-06}, {1.925e-03, 5.114e-04, 1.327e-03}},
            {{1.000e-08, 2.492e-06, 2.101e-06}, {1.032e-03, 2.499e-04, 5.670e-04}},
            {{1.000e-08, 7.571e-07, 1.032e-06}, {9.193e-04, 8.701e-05, 1.658e-04}},
            {{1.000e-08, 1.040e-06, 1.307e-06}, {1.208e-03, 2.773e-05, 4.544e-05}},
        }}
    },
    // ISO 102400
    {
        {{2.763e-04, 1.000e-08, 5.632e-05, 1.000e-08}, {2.324e-02, 2.530e-02, 2.491e-02, 2.492e-02}},
        {{
            {{3.416e-04, 6.339e-05, 5.365e-05}, {6.797e-03, 2.545e-04, 3.741e-03}},
            {{3.558e-05, 1.744e-05, 2.873e-05}, {3.637e-03, 1.092e-03, 2.794e-03}},
            {{1.000e-08, 2.768e-06, 2.035e-06}, {1.652e-03, 5.720e-04, 1.284e-03}},
            {{1.000e-08, 6.270e-07, 8.442e-07}, {1.133e-03, 1.931e-04, 3.861e-04}},
            {{1.000e-08, 1.104e-06, 1.611e-06}, {1.372e-03, 5.371e-05, 1.010e-04}},
        }}
    },
}};
