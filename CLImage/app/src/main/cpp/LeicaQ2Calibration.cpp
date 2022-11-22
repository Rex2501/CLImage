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
class LeicaQ2Calibration : public CameraCalibration<levels> {
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
        } else if (iso >= 6400 && iso < 12500) {
            float a = (iso - 6100) / 6100;
            return lerp<levels>(NLFData[6], NLFData[7], a);
        } else if (iso >= 12500 && iso < 25000) {
            float a = (iso - 12500) / 12500;
            return lerp<levels>(NLFData[7], NLFData[8], a);
        } else /* if (iso >= 25000 && iso <= 50000) */ {
            float a = (iso - 25000) / 25000;
            return lerp<levels>(NLFData[8], NLFData[9], a);
        }
    }

    std::pair<float, std::array<DenoiseParameters, levels>> getDenoiseParameters(int iso) const override {
        const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(102400) - log2(100)), 0.0, 1.0);

        std::cout << "LeicaQ2 DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

        float lerp = 0.5 * std::lerp(0.125f, 2.0f, nlf_alpha);
        float lerp_c = std::lerp(0.5f, 2.0f, nlf_alpha);

        // Default Good
        float lmult[5] = { 0.25, 1, 0.25, 0.125, 0.125 / 2 };
        float cmult[5] = { 1, 1, 0.5, 0.25, 0.125 };

        float chromaBoost = 8;

        std::array<DenoiseParameters, 5> denoiseParameters = {{
            {
                .luma = lmult[0] * lerp,
                .chroma = cmult[0] * lerp_c,
                .chromaBoost = chromaBoost,
                .gradientBoost = 32,
                .sharpening = 1.2, // Sharpen HF in LTM
            },
            {
                .luma = lmult[1] * lerp,
                .chroma = cmult[1] * lerp_c,
                .chromaBoost = chromaBoost,
                .sharpening = 1.1
            },
            {
                .luma = lmult[2] * lerp,
                .chroma = cmult[2] * lerp_c,
                .chromaBoost = chromaBoost,
                .sharpening = 1
            },
            {
                .luma = lmult[3] * lerp,
                .chroma = cmult[3] * lerp_c,
                .chromaBoost = chromaBoost,
                .sharpening = 1
            },
            {
                .luma = lmult[4] * lerp,
                .chroma = cmult[4] * lerp_c,
                .chromaBoost = chromaBoost,
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
                .shadows = 0.7,
                .highlights = 1.3,
                .detail = { 1, 1.05, 1.3 }
            }
        };
    }

    void calibrate(RawConverter* rawConverter, const std::filesystem::path& input_dir) const override {
        std::array<CalibrationEntry, 10> calibration_files = {{
            { 100,   "L1010611.DNG", { 3440, 777, 1549, 1006 }, false },
            { 200,   "L1010614.DNG", { 3440, 777, 1549, 1006 }, false },
            { 400,   "L1010617.DNG", { 3440, 777, 1549, 1006 }, false },
            { 800,   "L1010620.DNG", { 3440, 777, 1549, 1006 }, false },
            { 1600,  "L1010623.DNG", { 3440, 777, 1549, 1006 }, false },
            { 3200,  "L1010626.DNG", { 3440, 777, 1549, 1006 }, false },
            { 6400,  "L1010629.DNG", { 3440, 777, 1549, 1006 }, false },
            { 12500, "L1010632.DNG", { 3440, 777, 1549, 1006 }, false },
            { 25000, "L1010635.DNG", { 3440, 777, 1549, 1006 }, false },
            { 50000, "L1010638.DNG", { 3440, 777, 1549, 1006 }, false },
        }};

        std::array<NoiseModel<5>, 10> noiseModel;

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

            const auto rgb_image = CameraCalibration<5>::calibrate(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
            rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal.png", /*skip_alpha=*/ true);

            noiseModel[i] = demosaicParameters.noiseModel;
        }

        std::cout << "// Canon EOR RP Calibration table:" << std::endl;
        dumpNoiseModel(calibration_files, noiseModel);
    }
};

std::unique_ptr<CameraCalibration<5>> getLeicaQ2Calibration() {
    return std::make_unique<LeicaQ2Calibration<5>>();
}

void calibrateiLeicaQ2(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    LeicaQ2Calibration calibration;
    calibration.calibrate(rawConverter, input_dir);
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicLeicaQ2DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    LeicaQ2Calibration calibration;
    auto demosaicParameters = calibration.getDemosaicParameters(*inputImage, &dng_metadata, &exif_metadata);

    return RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, demosaicParameters.get(), /*calibrateFromImage=*/ false));
}

// --- NLFData ---

template<>
const std::array<NoiseModel<5>, 10> LeicaQ2Calibration<5>::NLFData = {{
    // ISO 100
    {
        {{6.998e-07, 1.030e-06, 1.000e-08, 1.034e-06}, {2.054e-04, 1.487e-04, 1.564e-04, 1.496e-04}},
        {{
            {{1.799e-06, 8.381e-08, 1.363e-07}, {1.126e-04, 5.149e-06, 9.256e-06}},
            {{1.216e-06, 1.331e-07, 1.399e-07}, {1.449e-04, 5.173e-06, 9.835e-06}},
            {{1.000e-08, 1.963e-07, 1.392e-07}, {3.046e-04, 5.064e-06, 9.020e-06}},
            {{1.000e-08, 3.895e-07, 3.640e-07}, {4.822e-04, 3.695e-06, 4.990e-06}},
            {{3.280e-06, 8.278e-07, 7.503e-07}, {8.459e-04, 3.336e-06, 4.069e-06}},
        }}
    },
    // ISO 200
    {
        {{1.191e-06, 1.083e-06, 1.759e-07, 1.085e-06}, {2.934e-04, 1.866e-04, 1.979e-04, 1.877e-04}},
        {{
            {{1.467e-06, 9.428e-08, 1.595e-07}, {1.451e-04, 7.415e-06, 1.352e-05}},
            {{5.287e-07, 1.272e-07, 1.289e-07}, {1.674e-04, 6.833e-06, 1.315e-05}},
            {{1.000e-08, 1.845e-07, 1.301e-07}, {3.147e-04, 5.711e-06, 1.013e-05}},
            {{1.000e-08, 3.817e-07, 3.526e-07}, {4.921e-04, 3.975e-06, 5.410e-06}},
            {{3.041e-06, 8.249e-07, 7.452e-07}, {8.473e-04, 3.471e-06, 4.177e-06}},
        }}
    },
    // ISO 400
    {
        {{3.958e-06, 1.725e-06, 1.034e-06, 1.796e-06}, {4.618e-04, 2.518e-04, 2.760e-04, 2.514e-04}},
        {{
            {{2.204e-06, 1.560e-07, 3.370e-07}, {1.748e-04, 1.123e-05, 2.002e-05}},
            {{1.854e-06, 1.854e-07, 2.580e-07}, {1.356e-04, 8.494e-06, 1.635e-05}},
            {{1.000e-08, 1.991e-07, 1.736e-07}, {2.961e-04, 6.421e-06, 1.128e-05}},
            {{1.000e-08, 3.837e-07, 3.489e-07}, {4.757e-04, 4.078e-06, 5.632e-06}},
            {{3.239e-06, 8.168e-07, 7.303e-07}, {8.478e-04, 3.457e-06, 4.195e-06}},
        }}
    },
    // ISO 800
    {
        {{2.363e-06, 1.214e-06, 4.945e-07, 1.225e-06}, {7.903e-04, 3.970e-04, 4.453e-04, 3.973e-04}},
        {{
            {{1.558e-06, 1.511e-07, 3.277e-07}, {2.689e-04, 1.924e-05, 3.380e-05}},
            {{2.717e-06, 1.867e-07, 2.462e-07}, {1.225e-04, 1.315e-05, 2.493e-05}},
            {{1.368e-06, 2.265e-07, 2.077e-07}, {2.392e-04, 7.270e-06, 1.317e-05}},
            {{1.000e-08, 3.802e-07, 3.550e-07}, {4.512e-04, 4.344e-06, 6.233e-06}},
            {{4.148e-06, 7.913e-07, 7.030e-07}, {8.511e-04, 3.671e-06, 4.395e-06}},
        }}
    },
    // ISO 1600
    {
        {{9.809e-06, 2.366e-06, 2.335e-06, 2.392e-06}, {1.393e-03, 6.724e-04, 7.647e-04, 6.734e-04}},
        {{
            {{1.233e-06, 2.787e-07, 5.380e-07}, {4.597e-04, 3.461e-05, 6.461e-05}},
            {{2.931e-06, 2.541e-07, 4.522e-07}, {1.410e-04, 2.271e-05, 4.236e-05}},
            {{1.524e-06, 2.574e-07, 2.653e-07}, {2.417e-04, 1.049e-05, 1.967e-05}},
            {{1.000e-08, 3.805e-07, 3.722e-07}, {4.583e-04, 5.347e-06, 7.898e-06}},
            {{3.902e-06, 7.645e-07, 6.842e-07}, {8.740e-04, 4.197e-06, 4.936e-06}},
        }}
    },
    // ISO 3200
    {
        {{3.768e-05, 5.565e-06, 8.192e-06, 5.303e-06}, {2.369e-03, 1.296e-03, 1.449e-03, 1.300e-03}},
        {{
            {{2.221e-06, 5.268e-07, 1.259e-06}, {5.725e-04, 4.954e-05, 9.645e-05}},
            {{2.481e-06, 4.588e-07, 1.015e-06}, {1.830e-04, 3.584e-05, 6.827e-05}},
            {{5.040e-07, 3.114e-07, 4.412e-07}, {2.801e-04, 1.765e-05, 3.258e-05}},
            {{1.000e-08, 3.950e-07, 4.219e-07}, {4.762e-04, 7.721e-06, 1.174e-05}},
            {{3.730e-06, 7.698e-07, 7.015e-07}, {8.663e-04, 4.620e-06, 5.787e-06}},
        }}
    },
    // ISO 6400
    {
        {{1.039e-04, 2.017e-05, 2.782e-05, 1.939e-05}, {4.812e-03, 2.575e-03, 2.917e-03, 2.570e-03}},
        {{
            {{1.006e-05, 1.720e-06, 4.254e-06}, {1.023e-03, 9.434e-05, 1.818e-04}},
            {{1.724e-06, 1.188e-06, 2.901e-06}, {3.120e-04, 6.891e-05, 1.319e-04}},
            {{1.000e-08, 5.880e-07, 1.150e-06}, {3.495e-04, 3.142e-05, 5.841e-05}},
            {{1.000e-08, 4.616e-07, 6.068e-07}, {5.282e-04, 1.238e-05, 1.995e-05}},
            {{2.550e-06, 7.758e-07, 7.540e-07}, {8.911e-04, 6.384e-06, 8.372e-06}},
        }}
    },
    // ISO 12500
    {
        {{2.195e-04, 5.161e-05, 7.401e-05, 5.177e-05}, {1.078e-02, 5.789e-03, 6.282e-03, 5.753e-03}},
        {{
            {{4.046e-05, 4.901e-06, 1.062e-05}, {1.699e-03, 1.817e-04, 3.807e-04}},
            {{6.159e-06, 3.386e-06, 7.547e-06}, {4.451e-04, 1.341e-04, 2.709e-04}},
            {{2.922e-06, 1.481e-06, 3.111e-06}, {2.998e-04, 5.573e-05, 1.086e-04}},
            {{1.000e-08, 7.120e-07, 1.114e-06}, {4.866e-04, 1.934e-05, 3.600e-05}},
            {{3.204e-06, 8.192e-07, 8.907e-07}, {9.186e-04, 8.184e-06, 1.244e-05}},
        }}
    },
    // ISO 25000
    {
        {{3.535e-04, 1.000e-08, 1.848e-05, 1.000e-08}, {1.635e-02, 1.747e-02, 1.901e-02, 1.724e-02}},
        {{
            {{1.245e-04, 1.038e-05, 2.442e-05}, {2.697e-03, 3.841e-04, 9.292e-04}},
            {{1.365e-05, 7.146e-06, 1.869e-05}, {8.140e-04, 2.908e-04, 6.195e-04}},
            {{3.276e-06, 3.055e-06, 7.791e-06}, {4.234e-04, 1.187e-04, 2.411e-04}},
            {{1.000e-08, 1.155e-06, 2.576e-06}, {5.546e-04, 3.762e-05, 7.448e-05}},
            {{1.000e-08, 8.494e-07, 1.311e-06}, {1.089e-03, 1.374e-05, 2.215e-05}},
        }}
    },
    // ISO 50000
    {
        {{3.935e-04, 8.987e-05, 1.992e-04, 9.729e-05}, {2.093e-02, 2.559e-02, 2.317e-02, 2.536e-02}},
        {{
            {{2.639e-04, 2.425e-05, 5.775e-05}, {5.571e-03, 9.186e-04, 2.664e-03}},
            {{2.772e-05, 1.394e-05, 4.626e-05}, {1.708e-03, 6.921e-04, 1.482e-03}},
            {{3.829e-06, 5.552e-06, 1.622e-05}, {7.120e-04, 2.864e-04, 6.102e-04}},
            {{1.000e-08, 1.865e-06, 5.662e-06}, {6.533e-04, 8.928e-05, 1.719e-04}},
            {{1.000e-08, 9.447e-07, 2.523e-06}, {1.270e-03, 2.686e-05, 4.426e-05}},
        }}
    },
}};
