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

static const std::array<NoiseModel, 10> LeicaQ2 = {{
    // ISO 100
    {
        { 4.2e-05, 5.1e-05, 4.9e-05, 5.1e-05 },
        {
            7.6e-06, 4.4e-07, 7.1e-07, 7.9e-05, 7.7e-06, 1.3e-05,
            6.5e-06, 4.4e-07, 5.5e-07, 1.6e-04, 6.9e-06, 1.3e-05,
            7.2e-06, 6.1e-07, 6.2e-07, 3.2e-04, 5.4e-06, 9.1e-06,
            5.2e-06, 1.1e-06, 1.0e-06, 7.8e-04, 4.0e-06, 6.2e-06,
            3.8e-05, 3.1e-06, 3.0e-06, 1.7e-03, 1.0e-06, 4.2e-06,
        },
    },
    // ISO 200
    {
        { 7.5e-05, 8.9e-05, 8.3e-05, 8.7e-05 },
        {
            7.2e-06, 4.5e-07, 7.4e-07, 1.2e-04, 1.4e-05, 2.4e-05,
            6.2e-06, 4.5e-07, 5.6e-07, 1.8e-04, 9.2e-06, 1.7e-05,
            5.2e-06, 5.7e-07, 5.8e-07, 3.8e-04, 7.1e-06, 1.2e-05,
            4.8e-06, 1.1e-06, 1.0e-06, 7.9e-04, 4.4e-06, 6.6e-06,
            3.7e-05, 3.1e-06, 3.0e-06, 1.7e-03, 9.6e-07, 4.4e-06,
        },
    },
    // ISO 400
    {
        { 1.4e-04, 1.6e-04, 1.6e-04, 1.5e-04 },
        {
            8.2e-06, 5.9e-07, 1.1e-06, 1.5e-04, 2.3e-05, 4.1e-05,
            8.5e-06, 5.6e-07, 7.9e-07, 1.2e-04, 1.1e-05, 2.1e-05,
            7.8e-06, 6.3e-07, 6.7e-07, 3.1e-04, 7.0e-06, 1.2e-05,
            5.2e-06, 1.1e-06, 1.0e-06, 7.8e-04, 4.8e-06, 7.1e-06,
            3.8e-05, 3.0e-06, 3.0e-06, 1.8e-03, 1.3e-06, 4.8e-06,
        },
    },
    // ISO 800
    {
        { 2.8e-04, 3.0e-04, 2.9e-04, 3.1e-04 },
        {
            7.5e-06, 4.9e-07, 9.8e-07, 2.6e-04, 4.6e-05, 7.8e-05,
            1.0e-05, 5.8e-07, 8.4e-07, 7.6e-05, 1.8e-05, 3.2e-05,
            8.3e-06, 6.2e-07, 6.5e-07, 3.0e-04, 9.1e-06, 1.6e-05,
            6.3e-06, 1.1e-06, 1.0e-06, 7.8e-04, 5.1e-06, 7.6e-06,
            3.9e-05, 2.9e-06, 2.8e-06, 1.8e-03, 1.9e-06, 5.1e-06,
        },
    },
    // ISO 1600
    {
        { 5.6e-04, 5.7e-04, 5.6e-04, 5.7e-04 },
        {
            7.9e-06, 8.4e-07, 2.0e-06, 4.7e-04, 8.6e-05, 1.5e-04,
            1.0e-05, 7.0e-07, 1.2e-06, 1.0e-04, 3.3e-05, 5.9e-05,
            8.2e-06, 6.6e-07, 7.3e-07, 3.1e-04, 1.3e-05, 2.4e-05,
            6.0e-06, 1.1e-06, 1.0e-06, 8.1e-04, 6.3e-06, 9.7e-06,
            4.1e-05, 2.8e-06, 2.7e-06, 1.9e-03, 2.8e-06, 6.2e-06,
        },
    },
    // ISO 3200
    {
        { 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03 },
        {
            9.6e-06, 2.2e-06, 5.3e-06, 9.4e-04, 1.7e-04, 3.0e-04,
            9.5e-06, 1.1e-06, 2.2e-06, 2.0e-04, 6.6e-05, 1.2e-04,
            8.0e-06, 8.2e-07, 1.1e-06, 3.2e-04, 2.2e-05, 4.1e-05,
            5.7e-06, 1.1e-06, 1.1e-06, 8.0e-04, 8.9e-06, 1.4e-05,
            3.9e-05, 2.9e-06, 2.8e-06, 1.8e-03, 2.9e-06, 6.4e-06,
        },
    },
    // ISO 6400
    {
        { 2.2e-03, 2.2e-03, 2.2e-03, 2.1e-03 },
        {
            1.3e-05, 5.1e-06, 1.2e-05, 1.0e-03, 2.5e-04, 4.4e-04,
            8.5e-06, 2.4e-06, 5.5e-06, 3.3e-04, 1.2e-04, 2.1e-04,
            4.8e-06, 1.2e-06, 2.1e-06, 4.5e-04, 4.4e-05, 7.9e-05,
            4.0e-06, 1.2e-06, 1.5e-06, 8.4e-04, 1.6e-05, 2.5e-05,
            3.3e-05, 2.9e-06, 3.1e-06, 1.9e-03, 5.2e-06, 9.6e-06,
        },
    },
    // ISO 12500
    {
        { 4.2e-03, 4.7e-03, 4.5e-03, 4.5e-03 },
        {
            4.1e-05, 1.9e-05, 3.9e-05, 1.8e-03, 4.6e-04, 8.4e-04,
            1.6e-05, 7.6e-06, 1.6e-05, 4.7e-04, 2.2e-04, 4.1e-04,
            1.1e-05, 2.9e-06, 5.1e-06, 3.5e-04, 7.6e-05, 1.5e-04,
            5.0e-06, 1.7e-06, 2.3e-06, 8.7e-04, 2.4e-05, 4.4e-05,
            4.2e-05, 2.8e-06, 3.1e-06, 2.1e-03, 9.2e-06, 1.5e-05,
        },
    },
    // ISO 25000
    {
        { 7.9e-03, 8.6e-03, 8.1e-03, 8.7e-03 },
        {
            1.1e-04, 5.1e-05, 1.0e-04, 3.6e-03, 9.1e-04, 2.1e-03,
            2.7e-05, 1.8e-05, 4.1e-05, 9.2e-04, 4.5e-04, 8.7e-04,
            1.4e-05, 5.9e-06, 1.3e-05, 4.9e-04, 1.6e-04, 3.1e-04,
            3.6e-06, 2.3e-06, 4.4e-06, 9.7e-04, 4.9e-05, 9.1e-05,
            4.7e-05, 2.5e-06, 3.5e-06, 2.4e-03, 1.9e-05, 2.4e-05,
        },
    },
    // ISO 50000
    {
        { 1.9e-02, 1.7e-02, 1.6e-02, 1.7e-02 },
        {
            1.7e-04, 1.1e-04, 1.4e-04, 9.5e-03, 2.6e-03, 9.4e-03,
            5.4e-05, 3.8e-05, 1.0e-04, 1.9e-03, 1.1e-03, 2.3e-03,
            1.8e-05, 1.1e-05, 3.0e-05, 8.4e-04, 4.0e-04, 7.8e-04,
            2.6e-06, 3.5e-06, 9.7e-06, 1.1e-03, 1.2e-04, 2.2e-04,
            4.0e-05, 2.3e-06, 6.5e-06, 2.6e-03, 3.9e-05, 4.0e-05,
        },
    }
}};

template <int levels>
static std::pair<gls::Vector<4>, gls::Matrix<levels, 6>> nlfFromIso(const std::array<NoiseModel, 10>& NLFData, int iso) {
    iso = std::clamp(iso, 100, 50000);
    if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return std::pair(lerpRawNLF(NLFData[0].rawNlf, NLFData[1].rawNlf, a), lerpNLF<levels>(NLFData[0].pyramidNlf, NLFData[1].pyramidNlf, a));
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return std::pair(lerpRawNLF(NLFData[1].rawNlf, NLFData[2].rawNlf, a), lerpNLF<levels>(NLFData[1].pyramidNlf, NLFData[2].pyramidNlf, a));
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return std::pair(lerpRawNLF(NLFData[2].rawNlf, NLFData[3].rawNlf, a), lerpNLF<levels>(NLFData[2].pyramidNlf, NLFData[3].pyramidNlf, a));
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return std::pair(lerpRawNLF(NLFData[3].rawNlf, NLFData[4].rawNlf, a), lerpNLF<levels>(NLFData[3].pyramidNlf, NLFData[4].pyramidNlf, a));
    } else if (iso >= 1600 && iso < 3200) {
        float a = (iso - 1600) / 1600;
        return std::pair(lerpRawNLF(NLFData[4].rawNlf, NLFData[5].rawNlf, a), lerpNLF<levels>(NLFData[4].pyramidNlf, NLFData[5].pyramidNlf, a));
    } else if (iso >= 3200 && iso < 6400) {
        float a = (iso - 3200) / 3200;
        return std::pair(lerpRawNLF(NLFData[5].rawNlf, NLFData[6].rawNlf, a), lerpNLF<levels>(NLFData[5].pyramidNlf, NLFData[6].pyramidNlf, a));
    } else if (iso >= 6400 && iso < 12500) {
        float a = (iso - 6100) / 6100;
        return std::pair(lerpRawNLF(NLFData[6].rawNlf, NLFData[7].rawNlf, a), lerpNLF<levels>(NLFData[6].pyramidNlf, NLFData[7].pyramidNlf, a));
    } else if (iso >= 12500 && iso < 25000) {
        float a = (iso - 12500) / 12500;
        return std::pair(lerpRawNLF(NLFData[7].rawNlf, NLFData[8].rawNlf, a), lerpNLF<levels>(NLFData[7].pyramidNlf, NLFData[8].pyramidNlf, a));
    } else /* if (iso >= 25000 && iso <= 50000) */ {
        float a = (iso - 25000) / 25000;
        return std::pair(lerpRawNLF(NLFData[8].rawNlf, NLFData[9].rawNlf, a), lerpNLF<levels>(NLFData[8].pyramidNlf, NLFData[9].pyramidNlf, a));
    }
}

std::pair<float, std::array<DenoiseParameters, 5>> LeicaQ2DenoiseParameters(int iso) {
    const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(102400) - log2(100)), 0.0, 1.0);

    std::cout << "LeicaQ2DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

    float lerp = 0.5 * std::lerp(0.125f, 2.0f, nlf_alpha);
    float lerp_c = std::lerp(0.5f, 2.0f, nlf_alpha);

    // Default Good
    float lmult[5] = { 0.125, 1, 0.25, 0.125, 0.125 / 2 };
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

gls::image<gls::rgb_pixel>::unique_ptr calibrateLeicaQ2(RawConverter* rawConverter,
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

    const auto denoiseParameters = LeicaQ2DenoiseParameters(iso);
    demosaicParameters->noiseLevel = denoiseParameters.first;
    demosaicParameters->denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, /*rotate_180=*/ false));
}

void calibrateLeicaQ2(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

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

    std::array<NoiseModel, 10> noiseModel;

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

        const auto rgb_image = calibrateLeicaQ2(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_new_full_v1.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for LeicaQ2:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "// ISO " << calibration_files[i].iso << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicLeicaQ2DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
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

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIso<5>(LeicaQ2, iso);
    const auto denoiseParameters = LeicaQ2DenoiseParameters(iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
}
