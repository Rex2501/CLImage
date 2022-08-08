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

static const std::array<NoiseModel, 11> Sonya6400 = {{
    // ISO 100
    {
        { 2.8e-05, 3.3e-05, 2.9e-05, 3.3e-05 },
        {
            0.0e+00, 3.2e-08, 8.5e-08, 4.8e-04, 1.9e-05, 2.5e-05,
            0.0e+00, 8.2e-08, 1.4e-07, 7.5e-04, 1.3e-05, 1.4e-05,
            0.0e+00, 3.5e-07, 3.9e-07, 1.0e-03, 8.6e-06, 7.5e-06,
            9.6e-07, 1.1e-06, 1.2e-06, 1.3e-03, 3.6e-06, 3.3e-06,
            5.7e-05, 2.9e-06, 3.7e-06, 2.3e-03, 9.4e-06, 1.4e-05,
        },
    },
    // ISO 200
    {
        { 5.1e-05, 5.8e-05, 5.3e-05, 6.0e-05 },
        {
            0.0e+00, 9.0e-08, 1.4e-07, 5.1e-04, 2.6e-05, 4.1e-05,
            0.0e+00, 9.6e-08, 1.6e-07, 7.5e-04, 1.5e-05, 1.8e-05,
            0.0e+00, 3.5e-07, 4.0e-07, 1.0e-03, 9.3e-06, 8.8e-06,
            9.2e-07, 1.1e-06, 1.2e-06, 1.3e-03, 3.8e-06, 3.7e-06,
            5.7e-05, 3.0e-06, 3.8e-06, 2.3e-03, 9.5e-06, 1.4e-05,
        },
    },
    // ISO 400
    {
        { 9.6e-05, 1.1e-04, 1.1e-04, 1.1e-04 },
        {
            0.0e+00, 1.9e-07, 2.0e-07, 5.6e-04, 3.9e-05, 7.2e-05,
            0.0e+00, 1.3e-07, 1.7e-07, 7.7e-04, 2.0e-05, 2.9e-05,
            0.0e+00, 3.6e-07, 4.1e-07, 1.0e-03, 1.1e-05, 1.2e-05,
            5.9e-07, 1.1e-06, 1.2e-06, 1.3e-03, 4.4e-06, 4.4e-06,
            5.7e-05, 3.0e-06, 3.8e-06, 2.3e-03, 9.3e-06, 1.4e-05,
        },
    },
    // ISO 800
    {
        { 2.0e-04, 2.1e-04, 2.1e-04, 2.2e-04 },
        {
            0.0e+00, 5.2e-07, 6.5e-07, 6.4e-04, 6.4e-05, 1.3e-04,
            0.0e+00, 2.1e-07, 2.6e-07, 7.7e-04, 2.9e-05, 4.8e-05,
            0.0e+00, 3.7e-07, 4.2e-07, 1.0e-03, 1.4e-05, 1.7e-05,
            2.8e-07, 1.1e-06, 1.2e-06, 1.3e-03, 5.2e-06, 5.8e-06,
            6.0e-05, 3.1e-06, 3.9e-06, 2.2e-03, 8.7e-06, 1.3e-05,
        },
    },
    // ISO 1600
    {
        { 4.2e-04, 4.1e-04, 3.8e-04, 4.0e-04 },
        {
            0.0e+00, 1.4e-06, 2.2e-06, 8.0e-04, 1.0e-04, 2.1e-04,
            0.0e+00, 4.3e-07, 6.2e-07, 8.1e-04, 4.4e-05, 8.0e-05,
            0.0e+00, 4.1e-07, 5.1e-07, 1.0e-03, 1.9e-05, 2.7e-05,
            5.4e-07, 1.1e-06, 1.2e-06, 1.3e-03, 6.5e-06, 8.2e-06,
            6.0e-05, 3.1e-06, 3.9e-06, 2.2e-03, 9.0e-06, 1.4e-05,
        },
    },
    // ISO 3200
    {
        { 7.7e-04, 7.8e-04, 8.1e-04, 8.0e-04 },
        {
            1.6e-06, 3.6e-06, 6.7e-06, 1.2e-03, 1.6e-04, 3.0e-04,
            0.0e+00, 1.0e-06, 1.7e-06, 8.4e-04, 7.0e-05, 1.3e-04,
            0.0e+00, 5.6e-07, 7.6e-07, 1.0e-03, 2.8e-05, 4.5e-05,
            4.2e-07, 1.2e-06, 1.4e-06, 1.3e-03, 9.8e-06, 1.2e-05,
            6.3e-05, 3.3e-06, 4.2e-06, 2.1e-03, 8.3e-06, 1.5e-05,
        },
    },
    // ISO 6400
    {
        { 1.5e-03, 1.6e-03, 1.5e-03, 1.6e-03 },
        {
            8.3e-06, 7.2e-06, 1.3e-05, 1.9e-03, 2.2e-04, 4.5e-04,
            0.0e+00, 2.2e-06, 3.9e-06, 9.0e-04, 1.1e-04, 2.2e-04,
            0.0e+00, 8.7e-07, 1.3e-06, 9.6e-04, 4.5e-05, 7.6e-05,
            1.5e-06, 1.3e-06, 1.6e-06, 1.2e-03, 1.5e-05, 2.1e-05,
            6.5e-05, 3.6e-06, 4.7e-06, 1.9e-03, 8.5e-06, 1.7e-05,
        },
    },
    // ISO 12800
    {
        { 3.0e-03, 3.1e-03, 2.9e-03, 3.1e-03 },
        {
            2.0e-05, 1.2e-05, 1.8e-05, 3.7e-03, 3.3e-04, 8.3e-04,
            0.0e+00, 4.5e-06, 7.5e-06, 1.1e-03, 1.8e-04, 3.6e-04,
            0.0e+00, 1.5e-06, 2.3e-06, 1.1e-03, 7.8e-05, 1.4e-04,
            2.2e-06, 1.3e-06, 1.7e-06, 1.2e-03, 2.5e-05, 4.0e-05,
            6.7e-05, 3.4e-06, 4.3e-06, 2.0e-03, 1.2e-05, 2.2e-05,
        },
    },
    // ISO 25600
    {
        { 6.3e-03, 6.2e-03, 6.0e-03, 6.2e-03 },
        {
            5.3e-05, 1.7e-05, 2.2e-05, 7.3e-03, 4.9e-04, 1.3e-03,
            9.2e-06, 6.6e-06, 9.7e-06, 1.7e-03, 3.4e-04, 7.0e-04,
            0.0e+00, 2.3e-06, 3.4e-06, 1.1e-03, 1.5e-04, 2.6e-04,
            3.0e-06, 1.5e-06, 2.1e-06, 1.2e-03, 4.7e-05, 7.3e-05,
            6.8e-05, 3.4e-06, 4.5e-06, 1.9e-03, 1.6e-05, 2.6e-05,
        },
    },
    // ISO 51200
    {
        { 8.6e-03, 8.7e-03, 8.2e-03, 9.0e-03 },
        {
            1.2e-04, 3.5e-05, 3.7e-05, 9.9e-03, 5.8e-04, 2.1e-03,
            2.5e-05, 1.4e-05, 1.4e-05, 2.7e-03, 6.0e-04, 1.4e-03,
            2.6e-06, 4.9e-06, 4.9e-06, 1.3e-03, 2.9e-04, 5.8e-04,
            3.6e-06, 2.0e-06, 2.6e-06, 1.3e-03, 9.9e-05, 1.7e-04,
            7.7e-05, 3.4e-06, 4.8e-06, 1.8e-03, 3.4e-05, 5.7e-05,
        },
    },
    // ISO 102400
    {
        { 1.8e-02, 1.7e-02, 1.7e-02, 1.7e-02 },
        {
            3.5e-04, 8.5e-05, 7.4e-05, 1.3e-02, -1.9e-04, 4.2e-03,
            3.2e-05, 1.8e-05, 4.1e-06, 6.2e-03, 1.4e-03, 3.5e-03,
            2.8e-06, 4.5e-06, 2.0e-06, 2.2e-03, 7.1e-04, 1.4e-03,
            0.0e+00, 1.8e-06, 1.8e-06, 1.5e-03, 2.3e-04, 4.2e-04,
            8.5e-05, 3.6e-06, 6.5e-06, 1.5e-03, 6.4e-05, 1.1e-04,
        },
    },
}};

template <int levels>
static std::pair<gls::Vector<4>, gls::Matrix<levels, 6>> nlfFromIso(const std::array<NoiseModel, 11>& NLFData, int iso) {
    iso = std::clamp(iso, 100, 102400);
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
    } else if (iso >= 25600 && iso < 51200) {
        float a = (iso - 25600) / 25600;
        return std::pair(lerpRawNLF(NLFData[8].rawNlf, NLFData[9].rawNlf, a), lerpNLF<levels>(NLFData[8].pyramidNlf, NLFData[9].pyramidNlf, a));
    } else /* if (iso >= 51200 && iso <= 102400) */ {
        float a = (iso - 51200) / 51200;
        return std::pair(lerpRawNLF(NLFData[9].rawNlf, NLFData[10].rawNlf, a), lerpNLF<levels>(NLFData[9].pyramidNlf, NLFData[10].pyramidNlf, a));
    }
}

std::pair<float, std::array<DenoiseParameters, 5>> Sonya6400DenoiseParameters(int iso) {
    const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(102400) - log2(100)), 0.0, 1.0);

    std::cout << "Sonya6400DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

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

gls::image<gls::rgb_pixel>::unique_ptr calibrateSonya6400(RawConverter* rawConverter,
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

    const auto denoiseParameters = Sonya6400DenoiseParameters(iso);
    demosaicParameters->noiseLevel = denoiseParameters.first;
    demosaicParameters->denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, /*rotate_180=*/ false));
}

void calibrateSonya6400(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

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

    std::array<NoiseModel, 11> noiseModel;

    for (int i = 0; i < calibration_files.size(); i++) {
        auto& entry = calibration_files[i];
        const auto input_path = input_dir / entry.fileName;

        DemosaicParameters demosaicParameters = {
            .rgbConversionParameters = {
                .localToneMapping = false
            }
        };

        const auto rgb_image = calibrateSonya6400(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_high_noise.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for Sonya6400:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "// ISO " << calibration_files[i].iso << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicSonya6400DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.05,
            .saturation = 1.0,
            .toneCurveSlope = 3.5,
            .localToneMapping = false
        },
        .ltmParameters = {
            .eps = 0.01,
            .shadows = 1, // 0.7,
            .highlights = 1, // 1.5,
            .detail = { 1, 1.1, 1.3 }
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIso<5>(Sonya6400, iso);
    const auto denoiseParameters = Sonya6400DenoiseParameters(iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
}
