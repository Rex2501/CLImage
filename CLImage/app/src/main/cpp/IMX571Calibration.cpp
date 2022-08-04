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

static const std::array<NoiseModel, 8> NLF_IMX571 = {{
    // ISO 100
    {
        { 1.8e-04, 1.0e-04, 1.6e-04, 9.0e-05 },
        {
            4.5e-06, 6.8e-07, 7.0e-07, 4.5e-05, 4.1e-05, 3.6e-05,
            7.6e-06, 1.1e-06, 1.5e-06, 2.6e-05, 2.1e-05, 2.4e-05,
            1.7e-05, 2.1e-06, 2.9e-06, 3.2e-05, 1.0e-05, 2.3e-05,
            4.1e-05, 3.8e-06, 5.1e-06, 9.3e-05, 1.2e-05, 4.3e-05,
            7.0e-05, 5.0e-06, 4.7e-06, 4.0e-04, 4.5e-05, 1.3e-04,
        },
    },
    // ISO 200
    {
        { 3.2e-04, 1.7e-04, 2.8e-04, 1.4e-04 },
        {
            4.5e-06, 8.5e-07, 7.7e-07, 7.0e-05, 7.7e-05, 6.6e-05,
            7.5e-06, 1.1e-06, 1.5e-06, 3.3e-05, 3.6e-05, 3.7e-05,
            1.7e-05, 2.1e-06, 2.9e-06, 3.5e-05, 1.5e-05, 2.7e-05,
            4.1e-05, 3.8e-06, 5.1e-06, 9.4e-05, 1.4e-05, 4.4e-05,
            7.0e-05, 5.0e-06, 4.7e-06, 4.0e-04, 4.5e-05, 1.3e-04,
        },
    },
    // ISO 400
    {
        { 6.0e-04, 3.1e-04, 5.3e-04, 2.5e-04 },
        {
            4.5e-06, 1.1e-06, 8.7e-07, 1.2e-04, 1.5e-04, 1.3e-04,
            7.5e-06, 1.3e-06, 1.5e-06, 4.9e-05, 6.7e-05, 6.3e-05,
            1.7e-05, 2.2e-06, 2.9e-06, 3.9e-05, 2.5e-05, 3.5e-05,
            4.1e-05, 3.8e-06, 5.0e-06, 9.6e-05, 1.6e-05, 4.6e-05,
            7.0e-05, 4.9e-06, 4.6e-06, 4.0e-04, 4.6e-05, 1.3e-04,
        },
    },
    // ISO 800
    {
        { 1.1e-03, 6.1e-04, 1.0e-03, 4.8e-04 },
        {
            5.0e-06, 2.1e-06, 1.6e-06, 2.2e-04, 2.8e-04, 2.5e-04,
            8.3e-06, 1.8e-06, 2.0e-06, 7.8e-05, 1.2e-04, 1.2e-04,
            1.9e-05, 2.7e-06, 3.5e-06, 4.2e-05, 4.3e-05, 5.4e-05,
            4.4e-05, 4.4e-06, 5.6e-06, 1.1e-04, 2.2e-05, 5.7e-05,
            7.8e-05, 5.3e-06, 5.2e-06, 4.2e-04, 5.1e-05, 1.4e-04,
        },
    },
    // ISO 1600
    {
        { 2.3e-03, 1.2e-03, 2.0e-03, 9.2e-04 },
        {
            5.0e-06, 3.0e-06, 2.5e-06, 4.4e-04, 5.5e-04, 4.9e-04,
            8.5e-06, 2.5e-06, 2.6e-06, 1.4e-04, 2.4e-04, 2.2e-04,
            1.9e-05, 2.9e-06, 3.7e-06, 5.6e-05, 8.0e-05, 8.6e-05,
            4.4e-05, 4.5e-06, 5.6e-06, 1.1e-04, 3.2e-05, 6.5e-05,
            8.0e-05, 5.5e-06, 5.4e-06, 4.2e-04, 5.1e-05, 1.4e-04,
        },
    },
    // ISO 3200
    {
        { 4.6e-03, 2.3e-03, 3.9e-03, 1.8e-03 },
        {
            4.9e-06, 6.7e-06, 3.8e-06, 9.0e-04, 1.1e-03, 9.9e-04,
            8.4e-06, 3.8e-06, 4.0e-06, 2.8e-04, 4.9e-04, 4.4e-04,
            2.0e-05, 3.6e-06, 4.2e-06, 9.0e-05, 1.5e-04, 1.5e-04,
            4.4e-05, 4.8e-06, 5.8e-06, 1.2e-04, 5.2e-05, 8.2e-05,
            8.1e-05, 5.7e-06, 5.7e-06, 4.2e-04, 5.6e-05, 1.4e-04,
        },
    },
    // ISO 6400
    {
        { 9.5e-03, 4.7e-03, 8.3e-03, 3.6e-03 },
        {
            2.1e-05, 3.8e-05, 2.1e-05, 1.9e-03, 2.3e-03, 2.1e-03,
            1.2e-05, 1.7e-05, 1.2e-05, 6.1e-04, 1.0e-03, 9.3e-04,
            1.9e-05, 7.5e-06, 6.2e-06, 1.9e-04, 3.3e-04, 3.1e-04,
            4.1e-05, 5.2e-06, 5.1e-06, 1.4e-04, 9.8e-05, 1.3e-04,
            7.3e-05, 4.4e-06, 2.8e-06, 4.2e-04, 6.9e-05, 1.7e-04,
        },
    },
    // ISO 10000
    {
        { 1.7e-02, 7.5e-03, 1.5e-02, 5.7e-03 },
        {
            3.9e-05, 4.0e-05, 0.0e+00, 2.8e-03, 4.8e-03, 4.7e-03,
            1.9e-05, 2.1e-05, 0.0e+00, 8.7e-04, 2.2e-03, 2.1e-03,
            2.3e-05, 1.2e-05, 5.6e-06, 2.5e-04, 6.9e-04, 6.6e-04,
            4.3e-05, 7.3e-06, 6.9e-06, 1.9e-04, 2.0e-04, 2.5e-04,
            7.5e-05, 6.7e-06, 8.5e-06, 5.2e-04, 1.1e-04, 2.5e-04,
        },
    },
}};


template <int levels>
static std::pair<gls::Vector<4>, gls::Matrix<levels, 6>> nlfFromIso(const std::array<NoiseModel, 8>& NLFData, int iso) {
    iso = std::clamp(iso, 100, 6400);
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
    } else /* if (iso >= 6400 && iso < 10000) */ {
        float a = (iso - 6400) / 3600;
        return std::pair(lerpRawNLF(NLFData[6].rawNlf, NLFData[7].rawNlf, a), lerpNLF<levels>(NLFData[6].pyramidNlf, NLFData[7].pyramidNlf, a));
    }
}

std::pair<float, std::array<DenoiseParameters, 5>> IMX571DenoiseParameters(int iso) {
    const float nlf_alpha = std::clamp((log2(iso) - log2(100)) / (log2(51200) - log2(100)), 0.0, 1.0);

    std::cout << "IMX571DenoiseParameters nlf_alpha: " << nlf_alpha << ", ISO: " << iso << std::endl;

    float lerp = std::lerp(1.0f, 2.0f, nlf_alpha);
    float lerp_c = std::lerp(1.0f, 4.0f, nlf_alpha);

    float lmult[5] = { 0.125, 0.5, 0.25, 0.125, 0.0625 };
    float cmult[5] = { 1, 2, 2, 1, 1 };

    // Bilateral
    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = lmult[0], // * lerp,
            .chroma = cmult[0] * lerp_c,
            .sharpening = std::lerp(1.5f, 1.0f, nlf_alpha)
        },
        {
            .luma = lmult[1] * lerp,
            .chroma = cmult[1] * lerp_c,
            .sharpening = 1, // 1.1
        },
        {
            .luma = lmult[2] * lerp,
            .chroma = cmult[2] * lerp_c,
            .sharpening = 1
        },
        {
            .luma = lmult[3] * lerp,
            .chroma = cmult[3] * lerp_c,
            .sharpening = 1
        },
        {
            .luma = lmult[4] * lerp,
            .chroma = cmult[4] * lerp_c,
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

    auto result = RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, rotate_180));

//    dng_metadata[TIFFTAG_CFAPATTERN] = std::vector<uint8_t>{ 1, 0, 2, 1 };
//    exif_metadata[EXIFTAG_ISOSPEEDRATINGS] = std::vector<uint16_t>{ (uint16_t) iso };
//    inputImage->write_dng_file((input_path.parent_path() / input_path.stem()).string() + "_ok.dng", gls::JPEG, &dng_metadata, &exif_metadata);

    return result;
}

void calibrateIMX571(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

    std::array<CalibrationEntry, 1> calibration_files = {{
        { 100,   "2022-06-15-11-03-22-196.dng",   {2246, 803, 2734, 1762}, false },
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

        const auto rgb_image = calibrateIMX571DNG(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position, entry.rotated);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_rgb.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for IMX571:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "// ISO " << calibration_files[i].iso << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "{ " << std::setprecision(4) << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << std::setprecision(4) << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicIMX571DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.0,
            // .saturation = 0.9,
            .toneCurveSlope = 3.5,
            // .exposureBias = -0.8,
            .blacks = 0.15,
            .localToneMapping = true
        },
        .ltmParameters = {
            .eps = 0.01,
            .shadows = 0.75,
            .highlights = 1.3,
            .detail = { 1, 1.1, 1.5 }
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.2594, -0.5333, -0.1138, -0.1404, 0.9717, 0.1688, 0.0342, 0.0969, 0.4330 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.8930, 1.0000, 1 / 1.7007 } });

    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 2" });

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
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

//    float exposureCompensation = 0.5 * smoothstep(0.01, 0.1, highlights) + 0.7;
//    if (exposureCompensation > 0) {
//        demosaicParameters.rgbConversionParameters.exposureBias = -exposureCompensation;
//        demosaicParameters.ltmParameters.shadows += 0.3 * exposureCompensation;
//        std::cout << "exposureBias: " << -exposureCompensation << std::endl;
//    }

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(inputImage, &demosaicParameters, nullptr, /*rotate_180=*/ false));
    // return RawConverter::convertToRGBImage(*rawConverter->fastDemosaicImage(inputImage, demosaicParameters));
}
