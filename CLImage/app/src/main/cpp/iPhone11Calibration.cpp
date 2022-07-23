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

static const std::array<NoiseModel, 8> iPhone11 = {{
    // ISO 32
    {
        { 1.5e-04, 1.9e-04, 2.3e-04, 1.5e-04 },
        {
            2.0e-05, 8.3e-07, 2.1e-06, 1.4e-04, 2.4e-05, 2.0e-05,
            3.2e-05, 1.7e-06, 3.5e-06, 7.0e-05, 7.9e-06, 1.2e-06,
            4.2e-05, 2.2e-06, 3.3e-06, 6.7e-05, 0.0e+00, 0.0e+00,
            5.3e-05, 2.7e-06, 3.7e-06, 1.8e-04, 0.0e+00, 0.0e+00,
            8.9e-05, 4.5e-06, 4.8e-06, 3.3e-04, 0.0e+00, 0.0e+00,
        },
    },
    // ISO 64
    {
        { 2.7e-04, 2.7e-04, 3.7e-04, 2.4e-04 },
        {
            2.0e-05, 9.3e-07, 2.1e-06, 1.8e-04, 4.4e-05, 4.3e-05,
            3.1e-05, 1.7e-06, 3.6e-06, 8.7e-05, 1.8e-05, 1.3e-05,
            4.1e-05, 2.6e-06, 3.8e-06, 7.2e-05, 0.0e+00, 0.0e+00,
            5.3e-05, 2.8e-06, 3.8e-06, 1.9e-04, 0.0e+00, 0.0e+00,
            8.9e-05, 4.5e-06, 4.6e-06, 3.3e-04, 0.0e+00, 0.0e+00,
        },
    },
    // ISO 100
    {
        { 3.2e-04, 4.6e-04, 5.9e-04, 3.7e-04 },
        {
            2.0e-05, 1.1e-06, 2.3e-06, 2.5e-04, 6.6e-05, 6.8e-05,
            3.1e-05, 1.7e-06, 3.7e-06, 1.1e-04, 2.9e-05, 2.6e-05,
            4.1e-05, 2.7e-06, 4.4e-06, 7.8e-05, 2.9e-06, 0.0e+00,
            5.3e-05, 2.9e-06, 3.9e-06, 1.9e-04, 0.0e+00, 0.0e+00,
            9.1e-05, 4.6e-06, 4.8e-06, 3.2e-04, 0.0e+00, 0.0e+00,
        },
    },
    // ISO 200
    {
        { 6.6e-04, 7.0e-04, 1.0e-03, 5.9e-04 },
        {
            2.1e-05, 1.6e-06, 3.2e-06, 4.2e-04, 1.3e-04, 1.3e-04,
            3.1e-05, 2.1e-06, 4.0e-06, 1.5e-04, 6.1e-05, 6.0e-05,
            4.1e-05, 2.9e-06, 5.3e-06, 9.0e-05, 1.4e-05, 3.9e-06,
            5.4e-05, 3.6e-06, 4.4e-06, 2.0e-04, 1.2e-07, 0.0e+00,
            9.8e-05, 5.3e-06, 5.0e-06, 2.8e-04, 0.0e+00, 0.0e+00,
        },
    },
    // ISO 400
    {
        { 9.2e-04, 8.8e-04, 1.4e-03, 6.9e-04 },
        {
            2.6e-05, 3.3e-06, 6.4e-06, 8.1e-04, 2.3e-04, 2.6e-04,
            3.2e-05, 3.2e-06, 5.3e-06, 2.6e-04, 1.1e-04, 1.4e-04,
            4.2e-05, 3.2e-06, 5.9e-06, 1.1e-04, 3.0e-05, 2.9e-05,
            5.4e-05, 3.7e-06, 5.4e-06, 2.0e-04, 2.8e-06, 0.0e+00,
            9.2e-05, 5.0e-06, 5.0e-06, 3.3e-04, 0.0e+00, 0.0e+00,
        },
    },
    // ISO 800
    {
        { 9.3e-04, 6.9e-04, 1.1e-03, 6.5e-04 },
        {
            4.3e-05, 1.1e-05, 1.7e-05, 1.6e-03, 3.8e-04, 4.9e-04,
            3.8e-05, 7.1e-06, 1.2e-05, 4.3e-04, 1.9e-04, 2.6e-04,
            4.4e-05, 4.5e-06, 8.3e-06, 1.5e-04, 5.8e-05, 7.5e-05,
            5.5e-05, 4.0e-06, 7.8e-06, 2.0e-04, 9.5e-06, 0.0e+00,
            8.6e-05, 4.9e-06, 5.8e-06, 3.6e-04, 0.0e+00, 0.0e+00,
        },
    },
    // ISO 1600
    {
        { 9.1e-04, 7.6e-04, 1.0e-03, 7.0e-04 },
        {
            1.2e-04, 4.3e-05, 5.6e-05, 3.1e-03, 7.1e-04, 8.6e-04,
            5.7e-05, 2.1e-05, 3.2e-05, 7.7e-04, 4.1e-04, 4.9e-04,
            4.7e-05, 9.0e-06, 1.5e-05, 2.5e-04, 1.4e-04, 1.7e-04,
            5.4e-05, 5.0e-06, 1.0e-05, 2.0e-04, 3.4e-05, 2.1e-05,
            8.2e-05, 6.0e-06, 6.7e-06, 3.6e-04, 0.0e+00, 0.0e+00,
        },
    },
    // ISO 2500
    {
        { 7.7e-04, 7.0e-04, 1.3e-03, 6.8e-04 },
        {
            2.7e-04, 1.0e-04, 1.3e-04, 4.0e-03, 9.0e-04, 1.3e-03,
            9.2e-05, 4.9e-05, 6.8e-05, 1.2e-03, 6.0e-04, 7.6e-04,
            5.8e-05, 1.8e-05, 2.7e-05, 3.5e-04, 2.1e-04, 2.7e-04,
            5.7e-05, 7.7e-06, 1.3e-05, 2.4e-04, 5.4e-05, 5.2e-05,
            9.0e-05, 7.6e-06, 8.8e-06, 3.4e-04, 0.0e+00, 0.0e+00,
        },
    },
}};

template <int levels>
std::pair<gls::Vector<4>, gls::Matrix<levels, 6>> nlfFromIsoiPhone(const std::array<NoiseModel, 8>& NLFData, int iso) {
    iso = std::clamp(iso, 32, 2500);
    if (iso >= 32 && iso < 64) {
        float a = (iso - 32) / 32;
        return std::pair(lerpRawNLF(NLFData[0].rawNlf, NLFData[1].rawNlf, a), lerpNLF<levels>(NLFData[0].pyramidNlf, NLFData[1].pyramidNlf, a));
    } else if (iso >= 64 && iso < 100) {
        float a = (iso - 64) / 36;
        return std::pair(lerpRawNLF(NLFData[1].rawNlf, NLFData[2].rawNlf, a), lerpNLF<levels>(NLFData[1].pyramidNlf, NLFData[2].pyramidNlf, a));
    } else if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return std::pair(lerpRawNLF(NLFData[2].rawNlf, NLFData[3].rawNlf, a), lerpNLF<levels>(NLFData[2].pyramidNlf, NLFData[3].pyramidNlf, a));
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return std::pair(lerpRawNLF(NLFData[3].rawNlf, NLFData[4].rawNlf, a), lerpNLF<levels>(NLFData[3].pyramidNlf, NLFData[4].pyramidNlf, a));
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return std::pair(lerpRawNLF(NLFData[4].rawNlf, NLFData[5].rawNlf, a), lerpNLF<levels>(NLFData[4].pyramidNlf, NLFData[5].pyramidNlf, a));
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return std::pair(lerpRawNLF(NLFData[5].rawNlf, NLFData[6].rawNlf, a), lerpNLF<levels>(NLFData[5].pyramidNlf, NLFData[6].pyramidNlf, a));
    } /* else if (iso >= 1600 && iso < 2500) */ {
        float a = (iso - 1600) / 900;
        return std::pair(lerpRawNLF(NLFData[6].rawNlf, NLFData[7].rawNlf, a), lerpNLF<levels>(NLFData[6].pyramidNlf, NLFData[7].pyramidNlf, a));
    }
}

std::pair<float, std::array<DenoiseParameters, 5>> iPhone11DenoiseParameters(int iso) {
    const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(6400) - log2(100)), 0.0, 1.0);

    std::cout << "iPhone11DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

    float lerp = 0.75 * std::lerp(0.2f, 2.0f, nlf_alpha);
    float lerp_c = std::lerp(1.0f, 2.0f, nlf_alpha);

    // Default Good
    float lmult[5] = { 4, 2, 1, 0.25, 0.125 / 2 };
    float cmult[5] = { 1, 1, 1, 1, 1 };

    float chromaBoost = 4;

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = lmult[0] * lerp,
            .chroma = cmult[0] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 2,
            .sharpening = 1.5, // std::lerp(1.5f, 1.0f, nlf_alpha)
        },
        {
            .luma = lmult[1] * lerp,
            .chroma = cmult[1] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 2,
            .sharpening = 1.2
        },
        {
            .luma = lmult[2] * lerp,
            .chroma = cmult[2] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 2,
            .sharpening = 1
        },
        {
            .luma = lmult[3] * lerp,
            .chroma = cmult[3] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 2,
            .sharpening = 1
        },
        {
            .luma = lmult[4] * lerp,
            .chroma = cmult[4] * lerp_c,
            .chromaBoost = chromaBoost,
            .gradientBoost = 2,
            .sharpening = 1
        }
    }};

    return { nlf_alpha, denoiseParameters };
}

gls::image<gls::rgb_pixel>::unique_ptr demosaiciPhone11(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .blacks = 0.1,
            .localToneMapping = true
        },
        .ltmParameters = {
            .guidedFilterEps = 0.01,
            .shadows = 1.35,
            .highlights = 1,
            .detail = 1.0,
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIsoiPhone<5>(iPhone11, iso);
    const auto denoiseParameters = iPhone11DenoiseParameters(iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
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

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, /*rotate_180=*/ false));
}

void calibrateiPhone11(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

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

        const auto rgb_image = calibrateiPhone11(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_rawnr_rgb.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for iPhone 11:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "// ISO " << calibration_files[i].iso << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}
