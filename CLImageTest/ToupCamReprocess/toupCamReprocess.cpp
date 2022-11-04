//
//  main.cpp
//  ToupCamReprocess
//
//  Created by Fabio Riccardi on 11/3/22.
//

#include <iostream>
#include <filesystem>

#include "gls_image.hpp"
#include "gls_tiff_metadata.hpp"

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

int main(int argc, const char * argv[]) {
    const auto input_path = std::filesystem::path(argv[1]);

    std::cout << "Processing file: " << input_path << std::endl;

    const auto raw_data = gls::image<gls::luma_pixel_16>::read_png_file(input_path.string());

    /*
     The sensor data is rotated by 180 degrees (upside down) and flipped (mirrored)
     We save the sensor data in stright up form and apply the same rotation to the
     CFA pattern, i.e.: GRBG -> RGGB
     */

    rotate180AndFlipHorizontal(raw_data.get());

    gls::tiff_metadata dng_metadata, exif_metadata;

    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "ToupCam 1" });
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.2594, -0.5333, -0.1138, -0.1404, 0.9717, 0.1688, 0.0342, 0.0969, 0.4330 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.4522, 1.0000, 1 / 2.2875 } });
    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 0, 1, 1, 2 } });
    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xffff } });

    exif_metadata.insert({ EXIFTAG_ISOSPEEDRATINGS, std::vector<uint16_t>{ 100 } });

    auto dng_file = (input_path.parent_path() / input_path.stem()).string() + ".dng";
    raw_data->write_dng_file(dng_file, /*compression=*/ gls::JPEG, &dng_metadata, &exif_metadata);

    return 0;
}
