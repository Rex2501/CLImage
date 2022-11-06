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
#include "demosaic_cl.hpp"
#include "raw_converter.hpp"

#include <array>
#include <cmath>
#include <filesystem>

static const std::array<NoiseModel, 8> iPhone11 = {{
    // ISO 32
    {
        {{9.796e-06, 1.137e-05, 1.103e-05, 1.174e-05}, {3.332e-04, 1.098e-04, 3.853e-04, 1.303e-04}},
        {{
            {{1.623e-05, 6.756e-07, 1.270e-06}, {1.103e-04, 1.899e-05, 1.960e-05}},
            {{2.286e-05, 9.445e-07, 1.891e-06}, {1.754e-05, 1.073e-05, 9.517e-06}},
            {{2.991e-05, 1.613e-06, 2.287e-06}, {1.000e-08, 4.324e-07, 1.000e-08}},
            {{4.146e-05, 1.652e-06, 2.175e-06}, {1.000e-08, 1.000e-08, 1.000e-08}},
            {{1.275e-04, 4.848e-06, 5.893e-06}, {9.786e-05, 1.000e-08, 1.000e-08}},
        }}
    },
    // ISO 64
    {
        {{9.270e-06, 1.196e-05, 1.036e-05, 1.220e-05}, {6.137e-04, 1.885e-04, 7.384e-04, 2.273e-04}},
        {{
            {{1.605e-05, 5.858e-07, 1.617e-06}, {2.040e-04, 3.580e-05, 3.571e-05}},
            {{2.290e-05, 1.025e-06, 2.041e-06}, {3.848e-05, 2.094e-05, 2.165e-05}},
            {{3.082e-05, 1.652e-06, 2.823e-06}, {1.000e-08, 3.941e-06, 1.588e-06}},
            {{4.169e-05, 1.848e-06, 2.393e-06}, {1.000e-08, 1.000e-08, 1.000e-08}},
            {{1.290e-04, 5.081e-06, 6.012e-06}, {1.001e-04, 1.000e-08, 1.000e-08}},
        }}
    },
    // ISO 100
    {
        {{1.117e-05, 1.307e-05, 1.244e-05, 1.331e-05}, {9.216e-04, 2.728e-04, 1.124e-03, 3.314e-04}},
        {{
            {{1.505e-05, 4.073e-07, 1.776e-06}, {3.192e-04, 5.599e-05, 5.432e-05}},
            {{2.290e-05, 1.170e-06, 2.249e-06}, {6.071e-05, 3.170e-05, 3.456e-05}},
            {{3.117e-05, 1.647e-06, 2.868e-06}, {2.582e-06, 8.033e-06, 6.680e-06}},
            {{4.190e-05, 2.045e-06, 2.661e-06}, {1.000e-08, 1.000e-08, 1.000e-08}},
            {{1.289e-04, 5.063e-06, 6.034e-06}, {9.892e-05, 1.000e-08, 1.000e-08}},
        }}
    },
    // ISO 200
    {
        {{1.909e-05, 1.499e-05, 2.294e-05, 1.503e-05}, {1.713e-03, 5.256e-04, 2.094e-03, 6.433e-04}},
        {{
            {{1.300e-05, 3.699e-07, 1.839e-06}, {6.509e-04, 1.144e-04, 1.111e-04}},
            {{2.371e-05, 1.348e-06, 2.941e-06}, {1.159e-04, 6.498e-05, 6.653e-05}},
            {{3.126e-05, 1.849e-06, 3.083e-06}, {1.603e-05, 2.001e-05, 1.960e-05}},
            {{4.264e-05, 2.743e-06, 3.353e-06}, {1.000e-08, 1.530e-07, 1.000e-08}},
            {{1.291e-04, 5.774e-06, 6.181e-06}, {1.098e-04, 1.000e-08, 1.000e-08}},
        }}
    },
    // ISO 400
    {
        {{4.682e-05, 1.797e-05, 4.957e-05, 1.904e-05}, {3.189e-03, 1.075e-03, 3.595e-03, 1.282e-03}},
        {{
            {{1.459e-05, 1.256e-06, 3.239e-06}, {8.419e-04, 1.553e-04, 1.775e-04}},
            {{2.430e-05, 1.477e-06, 3.878e-06}, {2.004e-04, 1.076e-04, 1.236e-04}},
            {{3.188e-05, 2.275e-06, 3.726e-06}, {4.518e-05, 3.883e-05, 4.768e-05}},
            {{4.439e-05, 2.813e-06, 4.307e-06}, {1.000e-08, 6.379e-06, 4.662e-06}},
            {{1.282e-04, 5.653e-06, 6.499e-06}, {1.112e-04, 1.000e-08, 1.000e-08}},
        }}
    },
    // ISO 800
    {
        {{1.426e-04, 3.052e-05, 1.231e-04, 3.340e-05}, {5.629e-03, 2.240e-03, 6.191e-03, 2.636e-03}},
        {{
            {{3.117e-05, 4.513e-06, 8.335e-06}, {1.501e-03, 2.862e-04, 3.787e-04}},
            {{2.578e-05, 2.794e-06, 6.726e-06}, {4.623e-04, 2.135e-04, 2.840e-04}},
            {{3.405e-05, 3.317e-06, 6.186e-06}, {9.622e-05, 7.431e-05, 1.025e-04}},
            {{4.744e-05, 3.150e-06, 5.325e-06}, {1.000e-08, 1.720e-05, 2.148e-05}},
            {{1.298e-04, 5.859e-06, 7.754e-06}, {1.072e-04, 1.000e-08, 1.000e-08}},
        }}
    },
    // ISO 1600
    {
        {{7.327e-05, 6.631e-06, 4.111e-05, 4.759e-06}, {1.831e-02, 7.874e-03, 2.095e-02, 9.035e-03}},
        {{
            {{1.197e-04, 1.689e-05, 2.225e-05}, {2.068e-03, 5.974e-04, 7.779e-04}},
            {{3.421e-05, 7.232e-06, 1.460e-05}, {9.670e-04, 4.954e-04, 6.138e-04}},
            {{3.794e-05, 4.244e-06, 9.483e-06}, {2.179e-04, 1.923e-04, 2.333e-04}},
            {{4.866e-05, 4.056e-06, 7.026e-06}, {2.909e-05, 4.964e-05, 5.925e-05}},
            {{1.280e-04, 7.246e-06, 9.486e-06}, {7.534e-05, 2.488e-06, 1.000e-08}},
        }}
    },
    // ISO 2500
    {
        {{6.911e-05, 1.000e-08, 1.029e-04, 1.000e-08}, {2.742e-02, 1.405e-02, 2.669e-02, 1.608e-02}},
        {{
            {{2.312e-04, 3.110e-05, 3.982e-05}, {2.292e-03, 8.744e-04, 1.285e-03}},
            {{5.482e-05, 1.533e-05, 2.580e-05}, {1.517e-03, 7.986e-04, 1.057e-03}},
            {{3.974e-05, 6.075e-06, 1.266e-05}, {4.477e-04, 3.274e-04, 4.335e-04}},
            {{4.892e-05, 5.760e-06, 9.669e-06}, {8.297e-05, 8.215e-05, 1.062e-04}},
            {{1.374e-04, 7.836e-06, 1.180e-05}, {5.162e-05, 8.733e-06, 2.078e-06}},
        }}
    },
}};

template <int levels>
NoiseModel nlfFromIsoiPhone(const std::array<NoiseModel, 8>& NLFData, int iso) {
    iso = std::clamp(iso, 32, 2500);
    if (iso >= 32 && iso < 64) {
        float a = (iso - 32) / 32;
        return lerp<levels>(NLFData[0], NLFData[1], a);
    } else if (iso >= 64 && iso < 100) {
        float a = (iso - 64) / 36;
        return lerp<levels>(NLFData[1], NLFData[2], a);
    } else if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return lerp<levels>(NLFData[2], NLFData[3], a);
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return lerp<levels>(NLFData[3], NLFData[4], a);
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return lerp<levels>(NLFData[4], NLFData[5], a);
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return lerp<levels>(NLFData[5], NLFData[6], a);
    } /* else if (iso >= 1600 && iso < 2500) */ {
        float a = (iso - 1600) / 900;
        return lerp<levels>(NLFData[6], NLFData[7], a);
    }
}

std::pair<float, std::array<DenoiseParameters, 5>> iPhone11DenoiseParameters(int iso) {
    const float nlf_alpha = std::clamp((log2(iso) - log2(32)) / (log2(2500) - log2(32)), 0.0, 1.0);

    std::cout << "iPhone11DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

    float lerp = std::lerp(0.125f, 1.2f, nlf_alpha);
    float lerp_c = std::lerp(0.5f, 1.2f, nlf_alpha);

    // Default Good
    float lmult[5] = { 0.125f, 1.0f, 0.5f, 0.25f, 0.125f };
    float cmult[5] = { 1, 1, 0.5f, 0.25f, 0.125f };

    float chromaBoost = std::lerp(4.0f, 8.0f, nlf_alpha);

    float gradientBoost = 1 + 2 * smoothstep(0.3, 0.6, nlf_alpha);

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = lmult[0] * lerp,
            .chroma = cmult[0] * lerp_c,
            .chromaBoost = 2 * chromaBoost,
            .gradientBoost = 8,
            .sharpening = std::lerp(1.5f, 1.0f, nlf_alpha)
        },
        {
            .luma = lmult[1] * lerp,
            .chroma = cmult[1] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = gradientBoost,
            .sharpening = 1.2
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

gls::image<gls::rgb_pixel>::unique_ptr demosaiciPhone11(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            // .exposureBias = 0.3,
//            .blacks = 0.1,
            .localToneMapping = false
        },
        .ltmParameters = {
            .eps = 0.01,
            .shadows = 1.0,
            .highlights = 1.5,
            .detail = { 1, 1.1, 1.5 }
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
//    const auto inputImageRGB = gls::image<gls::rgb_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);
//    // convertTosRGB(_glsContext, *inputImageRGB, localToneMapping->getMask(), clsRGBImage.get(), *demosaicParameters);
//    inputImageRGB->write_png_file("/Users/fabio/test.png");

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);
    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIsoiPhone<5>(iPhone11, iso);
    const auto denoiseParameters = iPhone11DenoiseParameters(iso);
    demosaicParameters.noiseModel = nlfParams;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, &demosaicParameters, /*calibrateFromImage=*/ true));
}

gls::image<gls::rgb_pixel>::unique_ptr calibrateiPhone11(RawConverter* rawConverter,
                                                         const std::filesystem::path& input_path,
                                                         DemosaicParameters* demosaicParameters,
                                                         int iso, const gls::rectangle& gmb_position) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false, &gmb_position, /*rotate_180=*/ false);

    // See if the ISO value is present and override
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    const auto denoiseParameters = iPhone11DenoiseParameters(iso);
    demosaicParameters->noiseLevel = denoiseParameters.first;
    demosaicParameters->denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, demosaicParameters, /*calibrateFromImage=*/ true));
}

void calibrateiPhone11(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    std::array<CalibrationEntry, 8> calibration_files = {{
        { 32,   "IPHONE11hSLI0032NRD.dng", { 1798, 2199, 382, 269 }, false },
        { 64,   "IPHONE11hSLI0064NRD.dng", { 1799, 2200, 382, 269 }, false },
        { 100,  "IPHONE11hSLI0100NRD.dng", { 1800, 2200, 382, 269 }, false },
        { 200,  "IPHONE11hSLI0200NRD.dng", { 1796, 2199, 382, 269 }, false },
        { 400,  "IPHONE11hSLI0400NRD.dng", { 1796, 2204, 382, 269 }, false },
        { 800,  "IPHONE11hSLI0800NRD.dng", { 1795, 2199, 382, 269 }, false },
        { 1600, "IPHONE11hSLI1600NRD.dng", { 1793, 2195, 382, 269 }, false },
        { 2500, "IPHONE11hSLI2500NRD.dng", { 1794, 2200, 382, 269 }, false }
    }};

    std::array<NoiseModel, 8> noiseModel;

    for (int i = 0; i < calibration_files.size(); i++) {
        auto& entry = calibration_files[i];
        const auto input_path = input_dir / entry.fileName;

        DemosaicParameters demosaicParameters = {
            .rgbConversionParameters = {
                .contrast = 1.05,
                .saturation = 1.0,
                .toneCurveSlope = 3.5,
            }
        };

        const auto rgb_image = calibrateiPhone11(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_rawnr_rgb.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "// iPhone 11 Calibration table:" << std::endl;
    dumpNoiseModel(calibration_files, noiseModel);
}
