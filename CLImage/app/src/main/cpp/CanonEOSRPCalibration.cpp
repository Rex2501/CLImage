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

template <size_t levels = 5>
class CanonEOSRPCalibration : public CameraCalibration<levels> {
    static const std::array<NoiseModel<levels>, 10> NLFData;

public:
    NoiseModel<levels> nlfFromIso(int iso) const override {
        iso = std::clamp(iso, 100, 50000);
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
        } else /* if (iso >= 25600 && iso <= 40000) */ {
            float a = (iso - 25600) / 15400;
            return lerp<levels>(NLFData[8], NLFData[9], a);
        }
    }

    std::pair<float, std::array<DenoiseParameters, levels>> getDenoiseParameters(int iso) const override {
        const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(102400) - log2(100)), 0.0, 1.0);

        std::cout << "CanonEOSRP DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

        float lerp = std::lerp(0.125f, 1.2f, nlf_alpha);
        float lerp_c = std::lerp(0.5f, 1.2f, nlf_alpha);

        // Default Good
        float lmult[5] = { 0.125f, 1.0f, 0.5f, 0.25f, 0.125f };
        float cmult[5] = { 1, 1, 1, 1, 1 };

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

    DemosaicParameters buildDemosaicParameters() const override {
        return {
            .rgbConversionParameters = {
                .contrast = 1.05,
                .saturation = 1.0,
                .toneCurveSlope = 3.5,
                .localToneMapping = false
            },
            .ltmParameters = {
                .eps = 0.01,
                .shadows = 1, // 0.5,
                .highlights = 1, // 1.5,
                .detail = { 1, 1.1, 1.3 }
            }
        };
    }

    void calibrate(RawConverter* rawConverter, const std::filesystem::path& input_dir) const override {
        std::array<CalibrationEntry, 10> calibration_files = {{
            { 100,   "IMG_1104_ISO_100.dng",   { 2541, 534, 1163, 758 }, false },
            { 200,   "IMG_1107_ISO_200.dng",   { 2541, 534, 1163, 758 }, false },
            { 400,   "IMG_1110_ISO_400.dng",   { 2541, 534, 1163, 758 }, false },
            { 800,   "IMG_1113_ISO_800.dng",   { 2541, 534, 1163, 758 }, false },
            { 1600,  "IMG_1116_ISO_1600.dng",  { 2541, 534, 1163, 758 }, false },
            { 3200,  "IMG_1119_ISO_3200.dng",  { 2541, 534, 1163, 758 }, false },
            { 6400,  "IMG_1122_ISO_6400.dng",  { 2541, 534, 1163, 758 }, false },
            { 12800, "IMG_1125_ISO_12800.dng", { 2541, 534, 1163, 758 }, false },
            { 25600, "IMG_1128_ISO_25600.dng", { 2541, 534, 1163, 758 }, false },
            { 40000, "IMG_1131_ISO_40000.dng", { 2541, 534, 1163, 758 }, false },
        }};

        std::array<NoiseModel<5>, 10> noiseModel;

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

        std::cout << "// Canon EOR RP Calibration table:" << std::endl;
        dumpNoiseModel(calibration_files, noiseModel);
    }
};

void calibrateCanonEOSRP(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    CanonEOSRPCalibration calibration;
    calibration.calibrate(rawConverter, input_dir);
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicCanonEOSRPDNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    CanonEOSRPCalibration calibration;
    auto demosaicParameters = calibration.getDemosaicParameters(*inputImage, &dng_metadata, &exif_metadata);

    return RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, demosaicParameters.get(), /*calibrateFromImage=*/ true));
}

// --- NLFData ---

template<>
const std::array<NoiseModel<5>, 10> CanonEOSRPCalibration<5>::NLFData = {{
    // ISO 100
    {
        {{3.283e-06, 2.362e-06, 1.682e-06, 2.393e-06}, {1.801e-04, 1.322e-04, 1.387e-04, 1.320e-04}},
        {{
            {{2.797e-06, 8.173e-08, 1.421e-07}, {1.181e-04, 3.569e-06, 4.932e-06}},
            {{1.157e-06, 1.034e-07, 1.215e-07}, {1.907e-04, 3.476e-06, 4.630e-06}},
            {{1.000e-08, 1.873e-07, 1.606e-07}, {3.599e-04, 2.678e-06, 3.155e-06}},
            {{1.000e-08, 3.464e-07, 3.457e-07}, {5.476e-04, 1.548e-06, 1.536e-06}},
            {{8.617e-06, 7.370e-07, 6.917e-07}, {1.045e-03, 2.143e-06, 1.893e-06}},
        }}
    },
    // ISO 200
    {
        {{3.408e-06, 2.233e-06, 1.704e-06, 2.264e-06}, {2.221e-04, 1.514e-04, 1.639e-04, 1.515e-04}},
        {{
            {{2.614e-06, 8.666e-08, 1.578e-07}, {1.339e-04, 4.790e-06, 6.940e-06}},
            {{9.126e-07, 1.081e-07, 1.281e-07}, {1.980e-04, 4.165e-06, 5.933e-06}},
            {{1.000e-08, 1.836e-07, 1.611e-07}, {3.648e-04, 2.921e-06, 3.573e-06}},
            {{1.000e-08, 3.435e-07, 3.404e-07}, {5.495e-04, 1.690e-06, 1.700e-06}},
            {{9.005e-06, 7.405e-07, 6.990e-07}, {1.037e-03, 2.136e-06, 1.811e-06}},
        }}
    },
    // ISO 400
    {
        {{1.000e-08, 1.000e-08, 1.000e-08, 1.000e-08}, {4.549e-04, 3.330e-04, 3.213e-04, 3.352e-04}},
        {{
            {{1.000e-08, 4.149e-08, 1.216e-07}, {3.622e-04, 9.477e-06, 1.389e-05}},
            {{1.000e-08, 1.866e-08, 5.607e-08}, {4.826e-04, 8.705e-06, 1.160e-05}},
            {{1.000e-08, 9.914e-08, 9.166e-08}, {7.298e-04, 6.135e-06, 7.029e-06}},
            {{1.000e-08, 2.868e-07, 2.795e-07}, {9.604e-04, 4.103e-06, 4.177e-06}},
            {{1.000e-08, 7.009e-07, 6.993e-07}, {1.379e-03, 5.942e-06, 4.644e-06}},
        }}
    },
    // ISO 800
    {
        {{7.480e-08, 1.000e-08, 1.000e-08, 1.000e-08}, {6.153e-04, 3.901e-04, 4.159e-04, 3.915e-04}},
        {{
            {{1.000e-08, 8.893e-08, 1.971e-07}, {3.998e-04, 1.385e-05, 2.167e-05}},
            {{1.000e-08, 4.207e-08, 8.742e-08}, {4.867e-04, 1.141e-05, 1.667e-05}},
            {{1.000e-08, 1.064e-07, 1.041e-07}, {7.281e-04, 7.139e-06, 8.782e-06}},
            {{1.000e-08, 2.803e-07, 2.783e-07}, {9.659e-04, 4.423e-06, 4.676e-06}},
            {{1.000e-08, 7.179e-07, 7.086e-07}, {1.368e-03, 5.582e-06, 4.605e-06}},
        }}
    },
    // ISO 1600
    {
        {{3.133e-06, 1.000e-08, 2.072e-07, 1.000e-08}, {9.422e-04, 5.147e-04, 6.210e-04, 5.156e-04}},
        {{
            {{1.000e-08, 2.104e-07, 4.288e-07}, {4.759e-04, 2.232e-05, 3.622e-05}},
            {{1.000e-08, 1.039e-07, 1.911e-07}, {5.004e-04, 1.676e-05, 2.671e-05}},
            {{1.000e-08, 1.277e-07, 1.326e-07}, {7.314e-04, 9.046e-06, 1.255e-05}},
            {{1.000e-08, 2.745e-07, 2.882e-07}, {9.690e-04, 5.119e-06, 5.683e-06}},
            {{1.000e-08, 7.131e-07, 7.046e-07}, {1.377e-03, 5.757e-06, 4.886e-06}},
        }}
    },
    // ISO 3200
    {
        {{1.209e-05, 3.372e-07, 3.669e-06, 1.815e-07}, {1.565e-03, 7.549e-04, 1.024e-03, 7.612e-04}},
        {{
            {{1.000e-08, 4.739e-07, 8.660e-07}, {6.341e-04, 3.939e-05, 6.706e-05}},
            {{1.000e-08, 2.668e-07, 4.759e-07}, {5.198e-04, 2.718e-05, 4.591e-05}},
            {{1.000e-08, 1.683e-07, 2.208e-07}, {7.371e-04, 1.319e-05, 1.959e-05}},
            {{1.000e-08, 2.868e-07, 3.072e-07}, {9.702e-04, 6.231e-06, 7.865e-06}},
            {{1.000e-08, 7.077e-07, 6.985e-07}, {1.387e-03, 6.145e-06, 5.754e-06}},
        }}
    },
    // ISO 6400
    {
        {{5.152e-05, 8.319e-06, 2.006e-05, 8.037e-06}, {2.196e-03, 1.078e-03, 1.533e-03, 1.086e-03}},
        {{
            {{4.366e-06, 8.977e-07, 1.839e-06}, {5.676e-04, 5.114e-05, 9.078e-05}},
            {{2.133e-06, 6.415e-07, 1.318e-06}, {2.602e-04, 3.794e-05, 6.629e-05}},
            {{1.000e-08, 3.474e-07, 5.311e-07}, {3.740e-04, 1.807e-05, 3.008e-05}},
            {{1.000e-08, 3.609e-07, 4.407e-07}, {5.571e-04, 6.630e-06, 9.487e-06}},
            {{8.765e-06, 7.391e-07, 7.180e-07}, {1.056e-03, 3.471e-06, 4.043e-06}},
        }}
    },
    // ISO 12800
    {
        {{1.106e-04, 1.855e-05, 4.755e-05, 1.766e-05}, {4.417e-03, 2.119e-03, 3.084e-03, 2.144e-03}},
        {{
            {{1.267e-05, 2.398e-06, 4.421e-06}, {9.705e-04, 9.682e-05, 1.814e-04}},
            {{2.871e-06, 1.555e-06, 2.965e-06}, {3.677e-04, 7.571e-05, 1.375e-04}},
            {{1.000e-08, 6.957e-07, 1.214e-06}, {3.980e-04, 3.438e-05, 5.864e-05}},
            {{1.000e-08, 4.379e-07, 6.060e-07}, {5.616e-04, 1.239e-05, 1.873e-05}},
            {{9.579e-06, 7.653e-07, 7.667e-07}, {1.051e-03, 4.840e-06, 6.471e-06}},
        }}
    },
    // ISO 25600
    {
        {{2.155e-04, 5.368e-05, 1.108e-04, 5.176e-05}, {9.895e-03, 4.630e-03, 6.654e-03, 4.675e-03}},
        {{
            {{4.887e-05, 7.306e-06, 1.215e-05}, {1.586e-03, 1.779e-04, 3.727e-04}},
            {{7.466e-06, 4.920e-06, 8.293e-06}, {5.855e-04, 1.533e-04, 2.964e-04}},
            {{9.727e-07, 1.985e-06, 3.239e-06}, {4.523e-04, 7.176e-05, 1.293e-04}},
            {{1.000e-08, 8.197e-07, 1.179e-06}, {5.850e-04, 2.396e-05, 4.089e-05}},
            {{9.214e-06, 8.467e-07, 9.109e-07}, {1.050e-03, 8.134e-06, 1.333e-05}},
        }}
    },
    // ISO 40000
    {
        {{2.173e-04, 4.227e-05, 8.064e-05, 4.117e-05}, {1.715e-02, 9.502e-03, 1.386e-02, 9.458e-03}},
        {{
            {{1.121e-04, 1.376e-05, 2.255e-05}, {1.934e-03, 2.518e-04, 5.797e-04}},
            {{1.580e-05, 9.002e-06, 1.615e-05}, {8.018e-04, 2.455e-04, 4.796e-04}},
            {{4.098e-06, 3.609e-06, 6.212e-06}, {4.919e-04, 1.206e-04, 2.247e-04}},
            {{8.729e-07, 1.275e-06, 2.197e-06}, {5.740e-04, 4.054e-05, 6.821e-05}},
            {{8.896e-06, 9.605e-07, 1.126e-06}, {1.084e-03, 1.283e-05, 2.189e-05}},
        }}
    },
}};
