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

#include "demosaic.hpp"
#include "raw_converter.hpp"

#include <array>
#include <cmath>
#include <filesystem>

static const std::array<NoiseModel, 11> NLF_IMX571 = {{
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
            {{1.000e-08, 3.341e-07, 6.274e-07}, {7.355e-04, 4.991e-05, 1.119e-04}},
            {{1.000e-08, 2.090e-07, 3.597e-07}, {5.348e-04, 4.119e-05, 8.415e-05}},
            {{1.000e-08, 1.243e-07, 1.349e-07}, {7.457e-04, 2.282e-05, 3.848e-05}},
            {{1.000e-08, 2.878e-07, 3.276e-07}, {9.170e-04, 1.014e-05, 1.329e-05}},
            {{1.000e-08, 8.197e-07, 8.623e-07}, {1.387e-03, 8.432e-06, 7.595e-06}},
        }}
    },
    // ISO 6400
    {
        {{4.137e-05, 1.000e-08, 4.311e-06, 1.000e-08}, {5.539e-03, 2.345e-03, 3.110e-03, 2.352e-03}},
        {{
            {{1.000e-08, 8.455e-07, 1.279e-06}, {1.185e-03, 9.089e-05, 2.219e-04}},
            {{1.000e-08, 5.327e-07, 8.838e-07}, {5.921e-04, 7.463e-05, 1.611e-04}},
            {{1.000e-08, 2.519e-07, 3.422e-07}, {7.514e-04, 3.785e-05, 6.937e-05}},
            {{1.000e-08, 3.212e-07, 3.736e-07}, {8.733e-04, 1.480e-05, 2.279e-05}},
            {{1.000e-08, 9.016e-07, 9.733e-07}, {1.240e-03, 8.445e-06, 8.934e-06}},
        }}
    },
    // ISO 12800
    {
        {{6.766e-05, 2.051e-06, 1.645e-05, 2.807e-06}, {1.142e-02, 4.548e-03, 6.092e-03, 4.585e-03}},
        {{
            {{1.223e-05, 2.376e-06, 3.516e-06}, {1.851e-03, 1.628e-04, 4.385e-04}},
            {{1.000e-08, 1.565e-06, 2.196e-06}, {7.661e-04, 1.375e-04, 3.242e-04}},
            {{1.000e-08, 5.940e-07, 8.404e-07}, {7.688e-04, 6.826e-05, 1.345e-04}},
            {{1.000e-08, 3.956e-07, 5.176e-07}, {9.036e-04, 2.481e-05, 4.223e-05}},
            {{1.000e-08, 8.896e-07, 9.831e-07}, {1.305e-03, 1.131e-05, 1.458e-05}},
        }}
    },
    // ISO 25600
    {
        {{1.579e-04, 1.000e-08, 1.000e-08, 3.245e-07}, {1.696e-02, 9.136e-03, 1.286e-02, 9.248e-03}},
        {{
            {{5.303e-05, 5.603e-06, 6.124e-06}, {3.034e-03, 3.044e-04, 9.005e-04}},
            {{1.000e-08, 3.296e-06, 3.904e-06}, {1.149e-03, 2.726e-04, 6.647e-04}},
            {{1.000e-08, 1.229e-06, 1.379e-06}, {8.270e-04, 1.321e-04, 2.748e-04}},
            {{1.000e-08, 6.522e-07, 8.282e-07}, {8.899e-04, 4.346e-05, 7.882e-05}},
            {{1.000e-08, 9.640e-07, 1.148e-06}, {1.232e-03, 1.640e-05, 2.286e-05}},
        }}
    },
    // ISO 51200
    {
        {{2.426e-04, 1.000e-08, 1.693e-06, 1.000e-08}, {1.800e-02, 1.339e-02, 1.708e-02, 1.357e-02}},
        {{
            {{1.419e-04, 1.359e-05, 1.243e-05}, {4.036e-03, 5.020e-04, 1.808e-03}},
            {{7.951e-06, 7.008e-06, 8.821e-06}, {1.783e-03, 5.022e-04, 1.299e-03}},
            {{1.000e-08, 2.720e-06, 2.610e-06}, {1.014e-03, 2.477e-04, 5.623e-04}},
            {{1.000e-08, 8.383e-07, 1.257e-06}, {9.205e-04, 8.711e-05, 1.646e-04}},
            {{1.000e-08, 1.070e-06, 1.377e-06}, {1.212e-03, 2.778e-05, 4.549e-05}},
        }}
    },
    // ISO 102400
    {
        {{2.763e-04, 1.000e-08, 5.632e-05, 1.000e-08}, {2.324e-02, 2.530e-02, 2.491e-02, 2.492e-02}},
        {{
            {{3.307e-04, 5.977e-05, 5.801e-05}, {6.235e-03, 3.553e-04, 3.563e-03}},
            {{3.489e-05, 1.801e-05, 3.339e-05}, {3.409e-03, 1.084e-03, 2.719e-03}},
            {{1.000e-08, 3.644e-06, 3.951e-06}, {1.634e-03, 5.695e-04, 1.281e-03}},
            {{1.000e-08, 1.006e-06, 1.585e-06}, {1.121e-03, 1.922e-04, 3.851e-04}},
            {{1.000e-08, 1.189e-06, 1.971e-06}, {1.365e-03, 5.397e-05, 9.966e-05}},
        }}
    },
}};

template <int levels>
static NoiseModel nlfFromIso(const std::array<NoiseModel, 11>& NLFData, int iso) {
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

std::pair<float, std::array<DenoiseParameters, 5>> IMX571DenoiseParameters(int iso) {
    const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(102400) - log2(100)), 0.0, 1.0);

    std::cout << "Sonya6400DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

    float lerp = std::lerp(0.125f, 1.2f, nlf_alpha);
    float lerp_c = std::lerp(0.5f, 1.2f, nlf_alpha);

    // Default Good
    float highNoise = smoothstep(0.0, 0.6, nlf_alpha);
    float lmult[5] = {
        std::lerp(0.5f, 0.5f, highNoise),
        std::lerp(1.0f, 4.0f, highNoise),
        std::lerp(0.5f, 0.5f, highNoise),
        std::lerp(0.25f, 0.5f, highNoise),
        std::lerp(0.125f, 0.25f, highNoise),
    };
    float cmult[5] = { 1, 1, 1, 1, 1 };

    float chromaBoost = 4;

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = lmult[0] * lerp,
            .chroma = cmult[0] * lerp_c,
            .chromaBoost = 2 * chromaBoost,
            .gradientBoost = 8,
            .sharpening = std::lerp(1.5f, 0.8f, nlf_alpha)
        },
        {
            .luma = lmult[1] * lerp,
            .chroma = cmult[1] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 1,
            .sharpening = 1.1
        },
        {
            .luma = lmult[2] * lerp,
            .chroma = cmult[2] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 1,
            .sharpening = 1
        },
        {
            .luma = lmult[3] * lerp,
            .chroma = cmult[3] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 1,
            .sharpening = 1
        },
        {
            .luma = lmult[4] * lerp,
            .chroma = cmult[4] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 1,
            .sharpening = 1
        }
    }};

    return { nlf_alpha, denoiseParameters };
}

void rotate180AndFlipHorizontal(gls::image<gls::luma_pixel_16>* inputImage) {
    for (int y = 0; y < inputImage->height; y++) {
        for (int x = 0; x < inputImage->width / 2; x++) {
            const auto t = (*inputImage)[y][x];
            (*inputImage)[y][x] = (*inputImage)[y][inputImage->width - 1 - x];
            (*inputImage)[y][inputImage->width - 1 - x] = t;
        }
    }

    for (int x = 0; x < inputImage->width; x++) {
        for (int y = 0; y < inputImage->height / 2; y++) {
            const auto t = (*inputImage)[y][x];
            (*inputImage)[y][x] = (*inputImage)[inputImage->height - 1 - y][x];
            (*inputImage)[inputImage->height - 1 - y][x] = t;
        }
    }

    for (int y = 0; y < inputImage->height; y++) {
        for (int x = 0; x < inputImage->width / 2; x++) {
            const auto t = (*inputImage)[y][x];
            (*inputImage)[y][x] = (*inputImage)[y][inputImage->width - 1 - x];
            (*inputImage)[y][inputImage->width - 1 - x] = t;
        }
    }
}

gls::image<gls::rgb_pixel>::unique_ptr calibrateIMX571DNG(RawConverter* rawConverter, const std::filesystem::path& input_path,
                                                          DemosaicParameters* demosaicParameters, int iso,
                                                          const gls::rectangle& gmb_position, bool rotate_180) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 2" });

    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.2594, -0.5333, -0.1138, -0.1404, 0.9717, 0.1688, 0.0342, 0.0969, 0.4330 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.8930, 1.0000, 1 / 1.7007 } });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false, &gmb_position, rotate_180);

    // See if the ISO value is present and override
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    const auto denoiseParameters = IMX571DenoiseParameters(iso);
    demosaicParameters->noiseLevel = denoiseParameters.first;
    demosaicParameters->denoiseParameters = denoiseParameters.second;

    auto result = RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, demosaicParameters, /*calibrateFromImage=*/ true));

//    dng_metadata[TIFFTAG_CFAPATTERN] = std::vector<uint8_t>{ 1, 0, 2, 1 };
//    exif_metadata[EXIFTAG_ISOSPEEDRATINGS] = std::vector<uint16_t>{ (uint16_t) iso };
//    inputImage->write_dng_file((input_path.parent_path() / input_path.stem()).string() + "_ok.dng", gls::JPEG, &dng_metadata, &exif_metadata);

    return result;
}

//void calibrateIMX571(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
//    std::array<CalibrationEntry, 1> calibration_files = {{
//        { 100,   "2022-06-15-11-03-22-196.dng",   {2246, 803, 2734, 1762}, false },
//    }};
//
//    std::array<NoiseModel, 8> noiseModel;
//
//    for (int i = 0; i < calibration_files.size(); i++) {
//        auto& entry = calibration_files[i];
//        const auto input_path = input_dir / entry.fileName;
//
//        DemosaicParameters demosaicParameters = {
//            .rgbConversionParameters = {
//                .contrast = 1.05,
//                .saturation = 1.0,
//                .toneCurveSlope = 3.5,
//            }
//        };
//
//        const auto rgb_image = calibrateIMX571DNG(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position, entry.rotated);
//        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_rgb.png", /*skip_alpha=*/ true);
//
//        noiseModel[i] = demosaicParameters.noiseModel;
//    }
//
//    std::cout << "Calibration table for IMX571:" << std::endl;
//    dumpNoiseModel<1, 8>(calibration_files, noiseModel);
//}

gls::image<gls::rgb_pixel>::unique_ptr demosaicIMX571DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.05,
            .saturation = 1.0,
            .toneCurveSlope = 3.5,
            .localToneMapping = true
        },
        .ltmParameters = {
            .eps = 0.01,
            .shadows = 0.8,
            .highlights = 1.5,
            .detail = { 1, 1.2, 2.0 }
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
//    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.2594, -0.5333, -0.1138, -0.1404, 0.9717, 0.1688, 0.0342, 0.0969, 0.4330 } });
//    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.8930, 1.0000, 1 / 1.7007 } });
//
//    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
//    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 2" });

    auto fullInputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    // A crop size with dimensions multiples of 128 and ratio of exactly 3:2, for a total resolution of 16MP
    const gls::size imageSize = { 4992, 3328 };
    const gls::rectangle crop({(fullInputImage->width - imageSize.width) / 2, (fullInputImage->height - imageSize.height) / 2}, imageSize);

    auto inputImage = gls::image<gls::luma_pixel_16>(*fullInputImage,
                                                     (fullInputImage->width - imageSize.width) / 2,
                                                     (fullInputImage->height - imageSize.height) / 2,
                                                     imageSize.width, imageSize.height);

    float highlights = 0;
    unpackDNGMetadata(inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ true, /*gmb_position=*/ nullptr, /*rotate_180=*/ false, &highlights);
    std::cout << "highlights: " << highlights << std::endl;

    float iso = 100;
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    const auto nlfParams = nlfFromIso<5>(NLF_IMX571, iso);
    const auto denoiseParameters = IMX571DenoiseParameters(iso);
    demosaicParameters.noiseModel = nlfParams;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

//    float exposureCompensation = 0.5 * smoothstep(0.01, 0.1, highlights) + 0.7;
//    if (exposureCompensation > 0) {
//        demosaicParameters.rgbConversionParameters.exposureBias = -exposureCompensation;
//        demosaicParameters.ltmParameters.shadows += 0.3 * exposureCompensation;
//        std::cout << "exposureBias: " << -exposureCompensation << std::endl;
//    }

    return RawConverter::convertToRGBImage(*rawConverter->runPipeline(inputImage, &demosaicParameters, /*calibrateFromImage=*/ true));
    // return RawConverter::convertToRGBImage(*rawConverter->runFastPipeline(inputImage, demosaicParameters));
}
