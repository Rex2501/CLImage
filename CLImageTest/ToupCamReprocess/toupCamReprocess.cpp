//
//  main.cpp
//  ToupCamReprocess
//
//  Created by Fabio Riccardi on 11/3/22.
//

#include <iostream>
#include <filesystem>
#include <regex>
#include <chrono>
#include <ctime>

#include "demosaic.hpp"
#include "raw_converter.hpp"
#include "CameraCalibration.hpp"

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

std::tuple<std::string, float, uint32_t> parse_filename(const std::string& filename) {
    const std::regex filename_regex("\([0-9]+\)_\([0-9]+\)_\([0-9]+\)");
    std::smatch regex_match;
    if (std::regex_match(filename, regex_match, filename_regex)) {
        const time_t time = static_cast<time_t>(std::atoll(regex_match[1].str().c_str())) / 1000000000;
        const std::string timestamp = std::ctime(&time);
        const float exposure_time = std::atol(regex_match[2].str().c_str()) / 1000000.0;
        const uint32_t ISO = (uint32_t) std::atol(regex_match[3].str().c_str());

        return {
            timestamp.substr(0, timestamp.size() - 1), // Skip trailing '\n'
            exposure_time,
            ISO
        };
    }
    throw std::domain_error("Can't parse filename.");
}

void raw_png_to_dng(const std::filesystem::path& input_path) {
    const std::string filename = input_path.filename().stem();
    std::cout << "Processing file: " << filename << std::endl;

    const auto basic_metadata = parse_filename(filename);

    const auto timestamp = std::get<0>(basic_metadata);
    const auto exposure_time = std::get<1>(basic_metadata);
    const auto ISO = std::get<2>(basic_metadata);

    const auto raw_data = gls::image<gls::luma_pixel_16>::read_png_file(input_path.string());

    /*
     The sensor data is rotated by 180 degrees (upside down) and flipped (mirrored)
     We save the sensor data in stright up form and apply the same rotation to the
     CFA pattern, i.e.: GRBG -> BGGR
     */

    rotate180AndFlipHorizontal(raw_data.get());

#if 0 // Use Gretag Machbeth Cart Coordinates
    // const auto gmb_position = gls::rectangle { 2538, 314, 868, 485 };

    const auto gmb_position = gls::rectangle { 4944, 1698, 909, 489 };

    gls::Vector<3> pre_mul;
    gls::Matrix<3, 3> cam_xyz;
    estimateRawParameters(*raw_data, &cam_xyz, &pre_mul, 0.0, (float) 0xffff, BayerPattern::bggr, gmb_position, false);
    std::cout << "cam_xyz:\n" << cam_xyz << "\npre_mul: " << pre_mul[1] / pre_mul << std::endl;

    const auto cam_xyz_span = cam_xyz.span();
    std::vector<float> color_matrix(cam_xyz_span.begin(), cam_xyz_span.end());

    const auto inv_pre_mul = pre_mul[1] / pre_mul;

    std::vector<float> as_shot_neutral(inv_pre_mul.begin(), inv_pre_mul.end());

    // Obtain the rgb_cam matrix and pre_mul
    const auto rgb_cam = cam_xyz_coeff(&pre_mul, cam_xyz);
    std::cout << "rgb_cam:\n" << rgb_cam << "\npre_mul: " << pre_mul[1] / pre_mul << std::endl;
#else
//    std::vector<float> color_matrix = { 1.9283, -0.8810, -0.1134, -0.0432, 0.8803, 0.1629, -0.0006, 0.1300, 0.3198 };
//    std::vector<float> as_shot_neutral = { 1 / 1.2273, 1.0000, 1 / 2.1288 };

    std::vector<float> color_matrix = { 1.2594, -0.5333, -0.1138, -0.1404, 0.9717, 0.1688, 0.0342, 0.0969, 0.4330 };
    std::vector<float> as_shot_neutral = { 1 / 1.8930, 1.0000, 1 / 1.7007 };
#endif

    gls::tiff_metadata dng_metadata, exif_metadata;

    // Basic DNG image interpretation metadata
    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "ToupCam 1" });
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, color_matrix });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, as_shot_neutral });

    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 2, 1, 1, 0 } });
    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xffff } });

    // Basic EXIF metadata
    exif_metadata.insert({ EXIFTAG_RECOMMENDEDEXPOSUREINDEX, (uint32_t) ISO });
    exif_metadata.insert({ EXIFTAG_ISOSPEEDRATINGS, std::vector<uint16_t>{ (uint16_t) ISO } });
    exif_metadata.insert({ EXIFTAG_DATETIMEORIGINAL, timestamp});
    exif_metadata.insert({ EXIFTAG_DATETIMEDIGITIZED, timestamp});
    exif_metadata.insert({ EXIFTAG_EXPOSURETIME, exposure_time});

    auto dng_file = (input_path.parent_path() / input_path.stem()).string() + "_b.dng";
    raw_data->write_dng_file(dng_file, /*compression=*/ gls::JPEG, &dng_metadata, &exif_metadata);
}

gls::image<gls::rgb_pixel>::unique_ptr demosaic_raw_data(RawConverter* rawConverter, gls::image<gls::luma_pixel_16>* raw_data) {
    /*
     The sensor data is rotated by 180 degrees (upside down) and flipped (mirrored)
     We save the sensor data in stright up form and apply the same rotation to the
     CFA pattern, i.e.: GRBG -> BGGR
     */

    rotate180AndFlipHorizontal(raw_data);

    std::vector<float> color_matrix = { 1.2594, -0.5333, -0.1138, -0.1404, 0.9717, 0.1688, 0.0342, 0.0969, 0.4330 };
    std::vector<float> as_shot_neutral = { 1 / 1.8930, 1.0000, 1 / 1.7007 };

    gls::tiff_metadata dng_metadata, exif_metadata;

    // Basic DNG image interpretation metadata
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, color_matrix });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, as_shot_neutral });

    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 2, 1, 1, 0 } });
    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xffff } });

    // Basic EXIF metadata
    const auto ISO = 100; // Provide the actual ISO informations
    exif_metadata.insert({ EXIFTAG_ISOSPEEDRATINGS, std::vector<uint16_t>{ (uint16_t) ISO } });

    return demosaicSonya6400RawImage(rawConverter, &dng_metadata, &exif_metadata, *raw_data);
}

void raw_png_to_rgb_png(RawConverter* rawConverter, const std::filesystem::path& input_path, const std::filesystem::path& output_path) {
    const std::string filename = input_path.filename().stem();
    std::cout << "Processing file: " << filename << std::endl;

    const auto raw_data = gls::image<gls::luma_pixel_16>::read_png_file(input_path.string());

    const auto rgbImage = demosaic_raw_data(rawConverter, raw_data.get());

    rgbImage->write_png_file(output_path);
}

void processDirectory(std::filesystem::path input_path, std::filesystem::path output_path) {
    std::cout << "Processing Directory: " << input_path.filename() << std::endl;

    auto input_dir = std::filesystem::directory_entry(input_path).is_directory() ? input_path : input_path.parent_path();
    std::vector<std::filesystem::path> directory_listing;
    std::copy(std::filesystem::directory_iterator(input_dir), std::filesystem::directory_iterator(),
              std::back_inserter(directory_listing));
    std::sort(directory_listing.begin(), directory_listing.end());

    if (!directory_listing.empty()) {
        gls::OpenCLContext glsContext("");
        RawConverter rawConverter(&glsContext);

        for (const auto& directory_entry : directory_listing) {
            if (directory_entry.filename().string().starts_with(".")) {
                continue;
            }

            if (std::filesystem::directory_entry(directory_entry).is_regular_file()) {
                const auto extension = directory_entry.extension();
                if ((extension != ".png" && extension != ".PNG")) {
                    continue;
                }

                // raw_png_to_dng(directory_entry);

                const auto filename = directory_entry.filename().stem().string();
                const auto output_file = output_path / (filename + "_rgb.png");

                std::cout << "Converting " << directory_entry << " to " << output_file << std::endl;

                if (!exists(status(output_path))) {
                    create_directory(output_path);
                }

                raw_png_to_rgb_png(&rawConverter, directory_entry, output_path / (filename + "_rgb.png"));
            } else if (std::filesystem::directory_entry(directory_entry).is_directory()) {
                processDirectory(directory_entry, output_path / directory_entry.filename());
            }
        }
    }
}

int main(int argc, const char * argv[]) {
    if (argc != 3) {
        std::cerr << "Please provide an input path and an output path." << std::endl;
        return -1;
    }
    const auto input_path = std::filesystem::path(argv[1]);
    const auto output_path = std::filesystem::path(argv[2]);

    processDirectory(input_path, output_path);

    return 0;
}
