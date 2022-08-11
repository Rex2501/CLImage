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

static const std::array<NoiseModel, 10> CanonEOSRP = {{
    // ISO 100
    {
        { 2.3e-05, 3.2e-05, 2.9e-05, 3.1e-05 },
        {
            1.1e-05, 3.4e-07, 5.1e-07, 6.7e-05, 5.7e-06, 8.6e-06,
            8.7e-06, 3.3e-07, 3.7e-07, 2.1e-04, 4.3e-06, 5.9e-06,
            7.1e-06, 4.6e-07, 4.5e-07, 4.7e-04, 3.1e-06, 3.4e-06,
            1.5e-05, 9.7e-07, 1.0e-06, 8.7e-04, 1.7e-06, 9.4e-07,
            4.5e-05, 2.2e-06, 2.7e-06, 2.6e-03, 8.5e-06, 1.2e-05,
        },
    },
    // ISO 200
    {
        { 3.7e-05, 4.8e-05, 4.5e-05, 4.6e-05 },
        {
            1.1e-05, 3.5e-07, 5.5e-07, 8.3e-05, 8.7e-06, 1.4e-05,
            8.6e-06, 3.3e-07, 3.9e-07, 2.1e-04, 5.3e-06, 7.9e-06,
            7.0e-06, 4.5e-07, 4.6e-07, 4.7e-04, 3.5e-06, 4.1e-06,
            1.5e-05, 9.7e-07, 1.0e-06, 8.7e-04, 1.9e-06, 1.3e-06,
            4.6e-05, 2.2e-06, 2.8e-06, 2.6e-03, 8.4e-06, 1.2e-05,
        },
    },
    // ISO 400
    {
        { 6.0e-05, 7.2e-05, 7.1e-05, 7.1e-05 },
        {
            0.0e+00, 2.1e-07, 3.7e-07, 4.8e-04, 2.1e-05, 3.3e-05,
            0.0e+00, 1.5e-07, 2.2e-07, 7.0e-04, 1.4e-05, 1.9e-05,
            0.0e+00, 3.4e-07, 3.8e-07, 9.8e-04, 8.0e-06, 8.5e-06,
            0.0e+00, 1.0e-06, 1.0e-06, 1.3e-03, 2.7e-06, 2.6e-06,
            5.4e-05, 2.6e-06, 3.4e-06, 2.2e-03, 5.7e-06, 9.3e-06,
        },
    },
    // ISO 800
    {
        { 1.1e-04, 1.2e-04, 1.2e-04, 1.2e-04 },
        {
            0.0e+00, 3.1e-07, 4.9e-07, 5.3e-04, 3.2e-05, 5.5e-05,
            0.0e+00, 1.8e-07, 2.7e-07, 7.1e-04, 1.8e-05, 2.7e-05,
            0.0e+00, 3.5e-07, 4.0e-07, 9.9e-04, 9.4e-06, 1.1e-05,
            0.0e+00, 9.9e-07, 1.0e-06, 1.3e-03, 3.1e-06, 3.2e-06,
            5.4e-05, 2.6e-06, 3.4e-06, 2.2e-03, 5.8e-06, 9.3e-06,
        },
    },
    // ISO 1600
    {
        { 2.2e-04, 2.3e-04, 2.3e-04, 2.3e-04 },
        {
            0.0e+00, 5.7e-07, 9.0e-07, 6.3e-04, 5.5e-05, 9.8e-05,
            0.0e+00, 2.7e-07, 4.1e-07, 7.2e-04, 2.7e-05, 4.5e-05,
            0.0e+00, 3.7e-07, 4.3e-07, 1.0e-03, 1.2e-05, 1.6e-05,
            0.0e+00, 1.0e-06, 1.1e-06, 1.3e-03, 4.0e-06, 4.8e-06,
            5.5e-05, 2.6e-06, 3.3e-06, 2.2e-03, 6.0e-06, 1.0e-05,
        },
    },
    // ISO 3200
    {
        { 4.0e-04, 4.2e-04, 4.4e-04, 4.3e-04 },
        {
            2.2e-06, 1.3e-06, 2.1e-06, 8.1e-04, 1.0e-04, 1.8e-04,
            0.0e+00, 5.1e-07, 8.3e-07, 7.6e-04, 4.5e-05, 7.9e-05,
            0.0e+00, 4.3e-07, 5.5e-07, 1.0e-03, 1.8e-05, 2.7e-05,
            0.0e+00, 1.0e-06, 1.1e-06, 1.3e-03, 5.5e-06, 7.5e-06,
            5.4e-05, 2.6e-06, 3.4e-06, 2.2e-03, 6.5e-06, 1.1e-05,
        },
    },
    // ISO 6400
    {
        { 7.4e-04, 7.1e-04, 7.1e-04, 7.5e-04 },
        {
            1.7e-05, 3.7e-06, 7.2e-06, 9.4e-04, 1.7e-04, 3.1e-04,
            1.3e-05, 1.4e-06, 2.7e-06, 2.7e-04, 7.0e-05, 1.3e-04,
            8.6e-06, 7.7e-07, 1.0e-06, 4.8e-04, 2.3e-05, 4.2e-05,
            1.5e-05, 1.0e-06, 1.2e-06, 8.9e-04, 8.1e-06, 1.2e-05,
            4.2e-05, 2.2e-06, 2.8e-06, 2.7e-03, 1.1e-05, 1.6e-05,
        },
    },
    // ISO 12800
    {
        { 1.3e-03, 1.4e-03, 1.2e-03, 1.4e-03 },
        {
            3.4e-05, 1.1e-05, 1.9e-05, 1.7e-03, 3.0e-04, 5.9e-04,
            1.5e-05, 3.7e-06, 6.4e-06, 4.5e-04, 1.4e-04, 2.6e-04,
            9.6e-06, 1.4e-06, 1.9e-06, 5.1e-04, 4.7e-05, 8.7e-05,
            1.6e-05, 1.2e-06, 1.4e-06, 8.9e-04, 1.6e-05, 2.4e-05,
            4.2e-05, 2.2e-06, 2.9e-06, 2.7e-03, 1.4e-05, 1.9e-05,
        },
    },
    // ISO 25600
    {
        { 2.0e-03, 2.5e-03, 1.8e-03, 2.6e-03 },
        {
            9.1e-05, 3.6e-05, 5.4e-05, 3.2e-03, 5.1e-04, 1.3e-03,
            2.4e-05, 1.2e-05, 1.9e-05, 8.4e-04, 2.7e-04, 5.4e-04,
            1.2e-05, 3.6e-06, 5.2e-06, 5.9e-04, 9.7e-05, 1.9e-04,
            1.6e-05, 1.7e-06, 2.2e-06, 9.2e-04, 3.0e-05, 5.4e-05,
            4.1e-05, 2.3e-06, 3.2e-06, 2.7e-03, 1.8e-05, 2.5e-05,
        },
    },
    // ISO 40000
    {
        { 2.8e-03, 3.7e-03, 2.6e-03, 3.9e-03 },
        {
            1.5e-04, 6.5e-05, 8.8e-05, 5.4e-03, 8.3e-04, 2.5e-03,
            4.0e-05, 2.3e-05, 3.8e-05, 1.2e-03, 4.2e-04, 8.7e-04,
            1.8e-05, 6.5e-06, 1.0e-05, 6.3e-04, 1.7e-04, 3.2e-04,
            1.6e-05, 2.3e-06, 3.7e-06, 9.7e-04, 5.4e-05, 9.2e-05,
            3.7e-05, 2.3e-06, 3.3e-06, 2.9e-03, 2.7e-05, 3.4e-05,
        },
    },
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
    } else if (iso >= 6400 && iso < 12800) {
        float a = (iso - 6400) / 6400;
        return std::pair(lerpRawNLF(NLFData[6].rawNlf, NLFData[7].rawNlf, a), lerpNLF<levels>(NLFData[6].pyramidNlf, NLFData[7].pyramidNlf, a));
    } else if (iso >= 12800 && iso < 25600) {
        float a = (iso - 12800) / 12800;
        return std::pair(lerpRawNLF(NLFData[7].rawNlf, NLFData[8].rawNlf, a), lerpNLF<levels>(NLFData[7].pyramidNlf, NLFData[8].pyramidNlf, a));
    } else /* if (iso >= 25600 && iso <= 40000) */ {
        float a = (iso - 25600) / 15400;
        return std::pair(lerpRawNLF(NLFData[8].rawNlf, NLFData[9].rawNlf, a), lerpNLF<levels>(NLFData[8].pyramidNlf, NLFData[9].pyramidNlf, a));
    }
}

std::pair<float, std::array<DenoiseParameters, 5>> CanonEOSRPDenoiseParameters(int iso) {
    const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(102400) - log2(100)), 0.0, 1.0);

    std::cout << "CanonEOSRPDenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

    float lerp = std::lerp(0.125f, 1.2f, nlf_alpha);
    float lerp_c = std::lerp(0.5f, 1.2f, nlf_alpha);

    // Default Good
    float highNoise = smoothstep(0.0, 0.6, nlf_alpha);
    float lmult[5] = {
        std::lerp(0.25f, 0.5f, highNoise),
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


gls::image<gls::rgb_pixel>::unique_ptr calibrateCanonEOSRP(RawConverter* rawConverter,
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

    const auto denoiseParameters = CanonEOSRPDenoiseParameters(iso);
    demosaicParameters->noiseLevel = denoiseParameters.first;
    demosaicParameters->denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, /*rotate_180=*/ false));
}

void calibrateCanonEOSRP(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

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

    std::array<NoiseModel, 10> noiseModel;

    for (int i = 0; i < calibration_files.size(); i++) {
        auto& entry = calibration_files[i];
        const auto input_path = input_dir / entry.fileName;

        DemosaicParameters demosaicParameters = {
            .rgbConversionParameters = {
                .localToneMapping = false
            }
        };

        const auto rgb_image = calibrateCanonEOSRP(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_high_noise.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for CanonEOSRP:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "// ISO " << calibration_files[i].iso << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicCanonEOSRPDNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .localToneMapping = false
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIso<5>(CanonEOSRP, iso);
    const auto denoiseParameters = CanonEOSRPDenoiseParameters(iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
}
