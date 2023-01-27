//
//  main.cpp
//  MonitorCalibration
//
//  Created by Fabio Riccardi on 1/25/23.
//

#include <iostream>
#include <filesystem>
#include <regex>
#include <vector>
#include <fstream>
#include <iostream>

#include "gls_image.hpp"

int main(int argc, const char * argv[]) {
    auto input_path = std::filesystem::path(argv[1]);

    std::cout << "Processing Directory: " << input_path.filename() << std::endl;

    const auto input_dir = std::filesystem::directory_entry(input_path).is_directory() ? input_path : input_path.parent_path();

    const auto monitor_calibration_file = input_dir / "MonitorCalibration.dat";

    // Monitor calibration data
    gls::rgb_pixel_16 lut[256];

    if (!exists(status(monitor_calibration_file))) {
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
                if ((extension != ".png" && extension != ".PNG")) {
                    continue;
                }

                int captureNumber = 0;
                int pixelLevel = 0;
                int exposureIndex = 0;

                const auto filename = input_path.filename().stem().string();
                const std::regex filename_regex("\([0-9]+\)_calibration_greyscale_\([0-9]+\).\([0-9]\)_rgb");
                std::smatch regex_match;
                if (std::regex_match(filename, regex_match, filename_regex)) {
                    captureNumber = std::atoi(regex_match[1].str().c_str());
                    pixelLevel = std::atoi(regex_match[2].str().c_str());
                    exposureIndex = std::atoi(regex_match[3].str().c_str());
                }

                const auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input_path.string());

                gls::Vector<3> pixelValue = gls::Vector<3>::zeros();
                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        pixelValue += gls::Vector<3>((*inputImage)[inputImage->height/2 + y][inputImage->width/2 + x].v);
                    }
                }
                pixelValue /= 25.0f;
                lut[pixelLevel] = pixelValue / 256.0f;

                std::cout << "Processing file " << filename << ", captureNumber: " << captureNumber << ", pixelLevel: " << pixelLevel << ", exposureIndex: " << exposureIndex << std::endl;
            }
        }

        // Make sure lookup table data is monotonic
        {
            gls::rgb_pixel_16 last_value = lut[255];
            for (int i = 254; i >= 0; i--) {
                for (int c = 0; c < 3; c++) {
                    if (lut[i][c] > last_value[c]) {
                        lut[i][c] = last_value[c];
                    } else {
                        last_value[c] = lut[i][c];
                    }
                }
            }
        }

        // Save monitor calibration data to file
        std::fstream monitor_calibration { monitor_calibration_file, monitor_calibration.out };
        for (int i = 0; i < 256; i++) {
            monitor_calibration << i << " " << lut[i].red << " " << lut[i].green << " " << lut[i].blue << std::endl;
        }
    } else {
        std::ifstream monitor_calibration { monitor_calibration_file, std::ios::in };

        int index = 0, last_index = -1;
        int red, green, blue;
        while (monitor_calibration >> index >> red >> green >> blue) {
            // Perform some input data validation
            if ((index != last_index + 1) || (red < 0 || red > 255) || (green < 0 || green > 255) || (blue < 0 || blue > 255)) {
                std::cerr << "Couldn't read calibration file: " << monitor_calibration_file.string() << std::endl;
                return -1;
            }
            last_index = index;

            lut[index] = { (uint16_t) red, (uint16_t) green, (uint16_t) blue};
        }
        if (index != 255) {
            std::cerr << "Couldn't read calibration file: " << monitor_calibration_file.string() << std::endl;
            return -1;
        }
    }

    // Pretty print display+camera transfer function
    for (int i = 0; i < 256; i++) {
        std::cout << "level " << i << ":\t" << lut[i].red << ", " << lut[i].green << ", " << lut[i].blue << std::endl;
    }

    // Build inverse transfer function to linearize the display+camera response
    gls::rgb_pixel_16 inv_lut[256];
    std::memset(inv_lut, 0, sizeof(inv_lut));

    // Fill the inverse LUT with available data
    for (int i = 0; i < 256; i++) {
        const auto entry = lut[i];
        for (int c = 0; c < 3; c++) {
            inv_lut[entry[c]][c] = i;
        }
    }

    // Interpolate table data for missing values
    gls::rgb_pixel_16 last_value = {0, 0, 0};
    gls::rgb_pixel_16 last_index = {0, 0, 0};
    for (int i = 0; i < 256; i++) {
        auto entry = inv_lut[i];
        if (i == 255) {
            for (int c = 0; c < 3; c++) {
                if (entry[c] == 0) {
                    entry[c] = 255;
                }
            }
        }
        for (int c = 0; c < 3; c++) {
            if (entry[c] != 0) {
                if (i - last_index[c] > 1) {
                    float delta = (entry[c] - last_value[c]) / (float) (i - last_index[c]);
                    for (int j = 0; j < i - last_index[c]; j++) {
                        inv_lut[last_index[c] + j][c] = last_value[c] + delta * j;
                    }
                }
                last_index[c] = i;
                inv_lut[i][c] = last_value[c] = entry[c];
            }
        }
    }

    // Pretty print display+camera inverse transfer function
    for (int i = 0; i < 256; i++) {
        std::cout << "inv level " << i << ":\t" << inv_lut[i].red << ", " << inv_lut[i].green << ", " << inv_lut[i].blue << std::endl;
    }

    // Read image to transform

    // const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_rgb.png";

    const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_macbeth_calibration_captures/3/107976_calibration_MacbethSingle.3_rgb.png";

    const auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input);

    // Apply inverse transfer function to captured data to recover the original color values
    inputImage->apply([&inv_lut] (gls::rgb_pixel_16* p, int x, int y) {
        for (int c = 0; c < 3; c++) {
            const uint8_t val8 = (*p)[c] >> 8;

            if (val8 > 0) {
                // For input values greater than the minimum LUT value, interpolate the 8 bit LUT entry over 16 bit data
                const float ratio = inv_lut[val8][c] / ((*p)[c] / 256.0f);
                (*p)[c] = std::clamp((*p)[c] * ratio, 0.0f, (float) 0xffff);
            } else {
                // Oterwise just use the value from the LUT
                (*p)[c] = inv_lut[val8][c] << 8;
            }
        }
    });

    // Write out transformed image

//    inputImage->write_png_file("/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_linrearized.png");

    inputImage->write_png_file("/Users/fabio/Desktop/107976_calibration_MacbethSingle.3_linrearized.png");

    return 0;
}
