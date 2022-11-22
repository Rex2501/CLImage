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

template <size_t levels = 5>
class RicohGRIIICalibration : public CameraCalibration<levels> {
    static const std::array<NoiseModel<levels>, 7> NLFData;

public:
    NoiseModel<levels> nlfFromIso(int iso) const override {
        iso = std::clamp(iso, 100, 6400);
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
        } else /* if (iso >= 3200 && iso < 6400) */ {
            float a = (iso - 3200) / 3200;
            return lerp<levels>(NLFData[5], NLFData[6], a);
        }
    }

    std::pair<float, std::array<DenoiseParameters, levels>> getDenoiseParameters(int iso) const override {
        const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(6400) - log2(100)), 0.0, 1.0);

        std::cout << "RicohGRIIIDenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

        float lerp = std::lerp(0.125f, 1.2f, nlf_alpha);
        float lerp_c = std::lerp(0.5f, 1.2f, nlf_alpha);

        // Default Good
        float lmult[5] = { 0.125, 1, 0.5, 0.25, 0.125 };
        float cmult[5] = { 1, 1, 0.5, 0.25, 0.125 };

        float chromaBoost = 8;

        std::array<DenoiseParameters, 5> denoiseParameters = {{
            {
                .luma = lmult[0] * lerp,
                .chroma = cmult[0] * lerp_c,
                .chromaBoost = chromaBoost,
                .sharpening = std::lerp(1.3f, 1.0f, nlf_alpha)
            },
            {
                .luma = lmult[1] * lerp,
                .chroma = cmult[1] * lerp_c,
                .chromaBoost = chromaBoost,
                .sharpening = 1, // 1.1
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
                .localToneMapping = false
            },
            .ltmParameters = {
                .eps = 0.01,
                .shadows = 0.6,
                .highlights = 1.2,
                .detail = { 1, 1.05, 1.5 }
            }
        };
    }

    void calibrate(RawConverter* rawConverter, const std::filesystem::path& input_dir) const override {
        std::array<CalibrationEntry, 7> calibration_files = {{
            { 100,  "R0000914_ISO100.DNG",  { 2437, 506, 1123, 733 }, false },
            { 200,  "R0000917_ISO200.DNG",  { 2437, 506, 1123, 733 }, false },
            { 400,  "R0000920_ISO400.DNG",  { 2437, 506, 1123, 733 }, false },
            { 800,  "R0000923_ISO800.DNG",  { 2437, 506, 1123, 733 }, false },
            { 1600, "R0000926_ISO1600.DNG", { 2437, 506, 1123, 733 }, false },
            { 3200, "R0000929_ISO3200.DNG", { 2437, 506, 1123, 733 }, false },
            { 6400, "R0000932_ISO6400.DNG", { 2437, 506, 1123, 733 }, false },
        }};

        std::array<NoiseModel<5>, 7> noiseModel;

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

        std::cout << "// RicohGRIII Calibration table:" << std::endl;
        dumpNoiseModel(calibration_files, noiseModel);
    }
};

void calibrateRicohGRIII(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    RicohGRIIICalibration calibration;
    calibration.calibrate(rawConverter, input_dir);
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicRicohGRIIIDNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    RicohGRIIICalibration calibration;
    auto demosaicParameters = calibration.getDemosaicParameters(*inputImage, &dng_metadata, &exif_metadata);

    return RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, demosaicParameters.get(), /*calibrateFromImage=*/ true));
}

// --- NLFData ---

template<>
const std::array<NoiseModel<5>, 7> RicohGRIIICalibration<5>::NLFData = {{
    // ISO 100
    {
        {{1.038e-06, 1.520e-06, 6.685e-07, 1.474e-06}, {2.471e-04, 1.567e-04, 1.622e-04, 1.570e-04}},
        {{
            {{4.090e-07, 4.088e-08, 4.067e-08}, {1.728e-04, 6.495e-06, 9.137e-06}},
            {{1.000e-08, 6.513e-08, 5.563e-08}, {2.604e-04, 6.792e-06, 8.006e-06}},
            {{1.000e-08, 1.816e-07, 1.449e-07}, {4.299e-04, 5.170e-06, 4.954e-06}},
            {{1.000e-08, 3.603e-07, 3.471e-07}, {7.393e-04, 3.703e-06, 2.719e-06}},
            {{6.411e-06, 7.370e-07, 6.175e-07}, {1.514e-03, 5.357e-06, 3.448e-06}},
        }}
    },
    // ISO 200
    {
        {{6.774e-07, 1.522e-06, 6.695e-07, 1.492e-06}, {3.559e-04, 1.890e-04, 2.060e-04, 1.888e-04}},
        {{
            {{4.913e-07, 4.885e-08, 4.957e-08}, {1.924e-04, 8.555e-06, 1.394e-05}},
            {{1.000e-08, 6.647e-08, 5.844e-08}, {2.627e-04, 8.052e-06, 1.101e-05}},
            {{1.000e-08, 1.762e-07, 1.402e-07}, {4.286e-04, 5.636e-06, 6.113e-06}},
            {{1.000e-08, 3.603e-07, 3.406e-07}, {7.375e-04, 3.784e-06, 3.127e-06}},
            {{6.107e-06, 7.397e-07, 6.167e-07}, {1.523e-03, 5.281e-06, 3.601e-06}},
        }}
    },
    // ISO 400
    {
        {{1.000e-08, 1.539e-06, 2.484e-07, 1.444e-06}, {4.044e-04, 2.274e-04, 2.334e-04, 2.286e-04}},
        {{
            {{1.121e-07, 3.441e-08, 1.000e-08}, {2.233e-04, 9.187e-06, 1.554e-05}},
            {{1.000e-08, 4.889e-08, 1.000e-08}, {2.720e-04, 9.163e-06, 1.352e-05}},
            {{1.000e-08, 1.637e-07, 1.198e-07}, {4.488e-04, 6.700e-06, 8.075e-06}},
            {{1.000e-08, 3.551e-07, 3.377e-07}, {7.385e-04, 4.292e-06, 3.883e-06}},
            {{6.059e-06, 7.512e-07, 6.308e-07}, {1.476e-03, 5.381e-06, 3.771e-06}},
        }}
    },
    // ISO 800
    {
        {{1.000e-08, 1.589e-06, 1.000e-08, 1.515e-06}, {4.766e-04, 3.106e-04, 2.908e-04, 3.119e-04}},
        {{
            {{1.000e-08, 4.296e-08, 1.000e-08}, {2.753e-04, 9.690e-06, 1.680e-05}},
            {{1.000e-08, 3.193e-08, 1.000e-08}, {2.844e-04, 1.082e-05, 1.707e-05}},
            {{1.000e-08, 1.483e-07, 1.071e-07}, {4.690e-04, 8.413e-06, 1.145e-05}},
            {{1.000e-08, 3.524e-07, 3.401e-07}, {7.264e-04, 4.891e-06, 5.204e-06}},
            {{5.845e-06, 7.774e-07, 6.506e-07}, {1.418e-03, 5.329e-06, 4.005e-06}},
        }}
    },
    // ISO 1600
    {
        {{1.000e-08, 1.583e-06, 1.000e-08, 1.425e-06}, {7.822e-04, 5.526e-04, 5.005e-04, 5.573e-04}},
        {{
            {{1.000e-08, 5.347e-08, 1.000e-08}, {4.387e-04, 1.369e-05, 2.504e-05}},
            {{1.000e-08, 1.000e-08, 1.000e-08}, {3.128e-04, 1.317e-05, 2.094e-05}},
            {{1.000e-08, 1.093e-07, 3.429e-08}, {4.485e-04, 9.681e-06, 1.376e-05}},
            {{1.000e-08, 3.749e-07, 3.469e-07}, {6.509e-04, 5.415e-06, 7.139e-06}},
            {{4.570e-06, 8.712e-07, 7.275e-07}, {1.266e-03, 5.309e-06, 4.898e-06}},
        }}
    },
    // ISO 3200
    {
        {{1.000e-08, 2.810e-06, 1.000e-08, 2.834e-06}, {1.255e-03, 9.297e-04, 8.122e-04, 9.316e-04}},
        {{
            {{9.165e-08, 1.062e-07, 1.000e-08}, {6.671e-04, 2.087e-05, 4.058e-05}},
            {{1.000e-08, 1.000e-08, 1.000e-08}, {3.338e-04, 1.786e-05, 3.181e-05}},
            {{1.000e-08, 1.059e-07, 3.176e-08}, {4.731e-04, 1.311e-05, 2.093e-05}},
            {{1.000e-08, 3.411e-07, 3.604e-07}, {7.081e-04, 7.767e-06, 1.087e-05}},
            {{5.374e-06, 7.988e-07, 6.865e-07}, {1.392e-03, 5.982e-06, 6.196e-06}},
        }}
    },
    // ISO 6400
    {
        {{1.000e-08, 5.618e-06, 4.201e-06, 5.653e-06}, {2.028e-03, 2.034e-03, 1.683e-03, 2.040e-03}},
        {{
            {{2.432e-06, 8.597e-08, 1.000e-08}, {9.223e-04, 3.517e-05, 6.895e-05}},
            {{1.000e-08, 1.863e-08, 1.000e-08}, {4.041e-04, 2.811e-05, 5.327e-05}},
            {{1.000e-08, 8.511e-08, 1.000e-08}, {4.799e-04, 1.958e-05, 3.412e-05}},
            {{1.000e-08, 4.279e-07, 5.243e-07}, {7.042e-04, 1.071e-05, 1.869e-05}},
            {{3.600e-06, 9.221e-07, 8.745e-07}, {1.288e-03, 7.740e-06, 9.474e-06}},
        }}
    },
}};
