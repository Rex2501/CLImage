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

#include <filesystem>
#include <string>
#include <cmath>
#include <chrono>

#include "gls_logging.h"
#include "gls_image.hpp"

#include "raw_converter.hpp"

#include "demosaic.hpp"
#include "gls_tiff_metadata.hpp"

#include "gls_linalg.hpp"

#include "CameraCalibration.hpp"

static const char* TAG = "RawPipeline Test";

void copyMetadata(const gls::tiff_metadata& source, gls::tiff_metadata* destination, ttag_t tag) {
    const auto entry = source.find(tag);
    if (entry != source.end()) {
        destination->insert({ tag, entry->second });
    }
}

void saveStrippedDNG(const std::string& file_name, const gls::image<gls::luma_pixel_16>& inputImage, const gls::tiff_metadata& dng_metadata, const gls::tiff_metadata& exif_metadata) {
    gls::tiff_metadata my_exif_metadata;
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_FNUMBER);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_EXPOSUREPROGRAM);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_EXPOSURETIME);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_DATETIMEORIGINAL);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_DATETIMEDIGITIZED);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_OFFSETTIME);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_OFFSETTIMEORIGINAL);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_OFFSETTIMEDIGITIZED);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_COMPONENTSCONFIGURATION);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_SHUTTERSPEEDVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_APERTUREVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_EXPOSUREBIASVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_MAXAPERTUREVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_METERINGMODE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LIGHTSOURCE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_FLASH);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_FOCALLENGTH);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_COLORSPACE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_SENSINGMETHOD);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_WHITEBALANCE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_BODYSERIALNUMBER);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSSPECIFICATION);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSMAKE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSMODEL);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSSERIALNUMBER);

    // Write out a stripped DNG files with minimal metadata
    inputImage.write_dng_file(file_name, /*compression=*/ gls::JPEG, &dng_metadata, &my_exif_metadata);
}

void transcodeAdobeDNG(const std::filesystem::path& input_path) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.4955, -0.6760, -0.1453, -0.1341, 1.0072, 0.1269, -0.0647, 0.1987, 0.4304 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.73344, 1, 1 / 1.68018 } });
//    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
//    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 2, 1, 1, 0 } });
//    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
//    // dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0x3fff } });
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    auto output_file = (input_path.parent_path() / input_path.stem()).string() + "_fixed.dng";
    saveStrippedDNG(output_file, *inputImage, dng_metadata, exif_metadata);
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicPlainFile(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .localToneMapping = false
        },
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    // Minimal noise model
    demosaicParameters.noiseModel.rawNlf = {
        {0, 0, 0, 0}, {2.8482e-04, 3.6848e-04, 3.3079e-04, 3.2821e-04}
    };
    demosaicParameters.noiseModel.pyramidNlf = {{
        {{2.1e-04, 1.0e-08, 8.2e-06}, {2.6e-04, 3.9e-05, 9.0e-06}},
        {{1.6e-04, 1.0e-08, 1.6e-06}, {6.1e-04, 1.8e-04, 6.1e-05}},
        {{2.5e-04, 1.0e-08, 1.0e-08}, {6.0e-04, 3.6e-04, 1.3e-04}},
        {{4.2e-04, 1.0e-08, 1.0e-08}, {4.2e-04, 8.8e-04, 3.4e-04}},
        {{3.0e-04, 1.0e-08, 1.0e-08}, {1.0e-03, 1.8e-03, 4.2e-04}},
    }};

    // Minimal noise model
    float lmult[5] = { 0.125, 0.125, 0.125, 0.125, 0.125 };
    float cmult[5] = { 0.125, 0.125, 0.125, 0.125, 0.125 };

    demosaicParameters.denoiseParameters = std::array<DenoiseParameters, 5> {{
        {
            .luma = lmult[0],
            .chroma = cmult[0],
            .sharpening = 1
        },
        {
            .luma = lmult[1],
            .chroma = cmult[1],
            .sharpening = 1
        },
        {
            .luma = lmult[2],
            .chroma = cmult[2],
            .sharpening = 1
        },
        {
            .luma = lmult[3],
            .chroma = cmult[3],
            .sharpening = 1
        },
        {
            .luma = lmult[4],
            .chroma = cmult[4],
            .sharpening = 1
        }
    }};

    return RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, &demosaicParameters, /*calibrateFromImage=*/ true));
}

void processKodakSet(gls::OpenCLContext* glsContext, const std::filesystem::path& input_path) {
    auto input_dir = std::filesystem::path(input_path.parent_path());
    std::vector<std::filesystem::path> directory_listing;
    std::copy(std::filesystem::directory_iterator(input_dir), std::filesystem::directory_iterator(),
              std::back_inserter(directory_listing));
    std::sort(directory_listing.begin(), directory_listing.end());

    for (const auto& input_path : directory_listing) {
        RawConverter rawConverter(glsContext);

        const auto extension = input_path.extension();
        if ((extension != ".png") || input_path.filename().string().starts_with(".")) {
            continue;
        }

        LOG_INFO(TAG) << "Processing: " << input_path.filename() << std::endl;

        const auto rgb = gls::image<gls::rgb_pixel>::read_png_file(input_path);

        gls::image<gls::luma_pixel_16> bayer(rgb->width, rgb->height);

        auto offsets = bayerOffsets[grbg];

        enum { red = 0, green = 1, blue = 2, green2 = 3 };

        bayer.apply([&offsets, &rgb](gls::luma_pixel_16* p, int x, int y) {
            for (int c : { red, green, blue, green2 }) {
                if ((x & 1) == (offsets[c].x & 1) && (y & 1) == (offsets[c].y & 1)) {
                    switch (c) {
                        case red:
                            p->luma = 0xff * (*rgb)[y][x].red;
                            break;
                        case blue:
                            p->luma = 0xff * (*rgb)[y][x].blue;
                            break;
                        case green:
                        case green2:
                            p->luma = 0xff * (*rgb)[y][x].green;
                            break;
                    }
                    break;
                }
            }
        });

        gls::tiff_metadata dng_metadata;
        dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{
             3.2404542, -1.5371385, -0.4985314,
            -0.9692660,  1.8760108,  0.04160,
             0.0556434, -0.2040259,  1.05752
        }});
        dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1, 1, 1 } });
        dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
        dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 1, 0, 2, 1 } });
        dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
        dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xffff } });
        dng_metadata.insert({ TIFFTAG_PROFILETONECURVE, std::vector<float>{ 0, 0, 1, 1 } });

        auto dng_file = (input_path.parent_path() / input_path.stem()).string() + ".dng";
        bayer.write_dng_file(dng_file, /*compression=*/ gls::NONE, &dng_metadata);

        const auto demosaiced = demosaicPlainFile(&rawConverter, dng_file);

        auto demosaiced_png_file = (input_path.parent_path() / input_path.stem()).string() + "_demosaiced_6_corr.PNG";
        demosaiced->write_png_file(demosaiced_png_file);
    }
}

void demosaicFile(RawConverter* rawConverter, std::filesystem::path input_path) {
    LOG_INFO(TAG) << "Processing File: " << input_path.filename() << std::endl;

    // transcodeAdobeDNG(input_path);
    // const auto rgb_image = demosaicIMX571DNG(rawConverter, input_path);
    const auto rgb_image = demosaicSonya6400DNG(rawConverter, input_path);
    // const auto rgb_image = demosaicCanonEOSRPDNG(rawConverter, input_path);
    // const auto rgb_image = demosaiciPhone11(rawConverter, input_path);
    // const auto rgb_image = demosaicRicohGRIII2DNG(rawConverter, input_path);
    // const auto rgb_image = demosaicLeicaQ2DNG(rawConverter, input_path);
    // rgb_image->write_jpeg_file((input_path.parent_path() /*/ "Processed" */ / input_path.stem()).string() + "_new_sharp_white.jpg", 100);
    rgb_image->write_png_file((input_path.parent_path() /* / "Processed" */ / input_path.stem()).string() + "_rgb_c_ln_raw_denoise_k.png", /*skip_alpha=*/ true);
}

void demosaicDirectory(RawConverter* rawConverter, std::filesystem::path input_path) {
    LOG_INFO(TAG) << "Processing Directory: " << input_path.filename() << std::endl;

    auto input_dir = std::filesystem::directory_entry(input_path).is_directory() ? input_path : input_path.parent_path();
    std::vector<std::filesystem::path> directory_listing;
    std::copy(std::filesystem::directory_iterator(input_dir), std::filesystem::directory_iterator(),
              std::back_inserter(directory_listing));
    std::sort(directory_listing.begin(), directory_listing.end());

    for (const auto& input_path : directory_listing) {
        if (input_path.filename().string().starts_with(".")) {
            continue;
        }

        if (std::filesystem::directory_entry(input_path).is_regular_file()) {
            const auto extension = input_path.extension();
            if ((extension != ".dng" && extension != ".DNG")) {
                continue;
            }
            demosaicFile(rawConverter, input_path);
        } else if (std::filesystem::directory_entry(input_path).is_directory()) {
            demosaicDirectory(rawConverter, input_path);
        }
    }
}

int main(int argc, const char* argv[]) {
    printf("RawPipeline Test!\n");

    if (argc > 1) {
        gls::OpenCLContext glsContext("");
        RawConverter rawConverter(&glsContext);

        auto input_path = std::filesystem::path(argv[1]);

        // processKodakSet(&glsContext, input_path);

        // calibrateIMX571(&rawConverter, input_path.parent_path());
        // calibrateCanonEOSRP(&rawConverter, input_path.parent_path());
        // calibrateiPhone11(&rawConverter, input_path.parent_path());
        // calibrateRicohGRIII(&rawConverter, input_path.parent_path());
        // calibrateSonya6400(&rawConverter, input_path.parent_path());
        // calibrateLeicaQ2(&rawConverter, input_path.parent_path());

        // demosaicDirectory(&rawConverter, input_path);

        demosaicFile(&rawConverter, input_path);

//        {
//            gls::tiff_metadata dng_metadata, exif_metadata;
//            const auto rawImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);
//            auto cpu_image = demosaicImageCPU(*rawImage, &dng_metadata, false);
//            cpu_image->apply([](gls::rgb_pixel_16* p, int x, int y) {
//                *p = {
//                    (uint16_t) (0xffff * sqrt((*p)[0] / (float) 0xffff)),
//                    (uint16_t) (0xffff * sqrt((*p)[1] / (float) 0xffff)),
//                    (uint16_t) (0xffff * sqrt((*p)[2] / (float) 0xffff))
//                };
//            });
//            cpu_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cpu_rgb.png", /*skip_alpha=*/ true);
//        }
    }
}
