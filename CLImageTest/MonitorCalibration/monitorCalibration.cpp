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
#include <cmath>

#include "gls_image.hpp"

struct monitor_response {
    gls::point coordinates;
    std::array<gls::rgb_pixel_16, 256> data;
};

constexpr const int inset = 300;
std::array<std::array<gls::point, 3>, 3> computeCalibrationCoordinates(const gls::image<gls::rgb_pixel_16>& inputImage) {
    return {{
        {{{ inset, inset }, { inputImage.width / 2, inset }, { inputImage.width - inset, inset }}},
        {{{ inset, inputImage.height / 2 }, { inputImage.width / 2, inputImage.height / 2 }, { inputImage.width - inset, inputImage.height / 2 }}},
        {{{ inset, inputImage.height - inset }, { inputImage.width / 2, inputImage.height - inset }, { inputImage.width - inset, inputImage.height - inset }}}
    }};
}

void cantParseFile(const std::filesystem::path& p) {
    std::cerr << "Couldn't read calibration file: " << p.string() << std::endl;
}

gls::rgb_pixel_16 lookup(const std::array<gls::rgb_pixel_16, 256>& lut, const gls::rgb_pixel_16& val) {
    gls::rgb_pixel_16 result;
    for (int c = 0; c < 3; c++) {
        const uint8_t val8 = val[c] >> 8;

        if (val8 > 0) {
            // For input values greater than the minimum LUT value, interpolate the 8 bit LUT entry over 16 bit data
            const float ratio = lut[val8][c] / (val[c] / 256.0f);
            result[c] = std::clamp(val[c] * ratio, 0.0f, (float) 0xffff);
        } else {
            // Oterwise just use the value from the LUT
            result[c] = lut[val8][c] << 8;
        }
    }
    return result;
}

int main(int argc, const char * argv[]) {
    auto input_path = std::filesystem::path(argv[1]);

    std::cout << "Processing Directory: " << input_path.filename() << std::endl;

    const auto input_dir = std::filesystem::directory_entry(input_path).is_directory() ? input_path : input_path.parent_path();

    const auto monitor_calibration_file = input_dir / "MonitorCalibration.dat";

    const constexpr gls::size calibration_grid = {3, 3};
    std::array<std::array<monitor_response, calibration_grid.width>, calibration_grid.height> calibration_data;

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

                std::cout << "Processing file " << filename << ", captureNumber: " << captureNumber << ", pixelLevel: " << pixelLevel << ", exposureIndex: " << exposureIndex << std::endl;

                const auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input_path.string());

                const auto calibration_coordinates = computeCalibrationCoordinates(*inputImage);

                for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
                    for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                        const auto& p = calibration_data[cg_y][cg_x].coordinates = calibration_coordinates[cg_y][cg_x];
                        auto& lut = calibration_data[cg_y][cg_x].data;

                        gls::Vector<3> pixelValue = gls::Vector<3>::zeros();
                        for (int y = -2; y <= 2; y++) {
                            for (int x = -2; x <= 2; x++) {
                                pixelValue += gls::Vector<3>((*inputImage)[p.y + y][p.x + x].v);
                            }
                        }
                        pixelValue /= 25.0f;
                        lut[pixelLevel] = pixelValue / 256.0f;
                    }
                }
            }
        }

        // Make sure lookup table data is monotonic
        for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
            for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                auto& lut = calibration_data[cg_y][cg_x].data;
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
        }

        // Save monitor calibration data to file
        std::fstream monitor_calibration { monitor_calibration_file, monitor_calibration.out };
        monitor_calibration << calibration_grid.width << " " << calibration_grid.height << std::endl;
        for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
            for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                monitor_calibration << calibration_data[cg_y][cg_x].coordinates.x << " " << calibration_data[cg_y][cg_x].coordinates.y << std::endl;

                const auto& lut = calibration_data[cg_y][cg_x].data;
                for (int i = 0; i < 256; i++) {
                    monitor_calibration << i << " " << lut[i].red << " " << lut[i].green << " " << lut[i].blue << std::endl;
                }
            }
        }
    } else {
        std::ifstream monitor_calibration { monitor_calibration_file, std::ios::in };

        gls::size grid;
        if (!(monitor_calibration >> grid.width >> grid.height)) {
            cantParseFile(monitor_calibration_file);
            return -1;
        }

        assert( grid == calibration_grid );

        for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
            for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                std::cout << "Reading grid element " << cg_x << ", " << cg_y << std::endl;

                if (!(monitor_calibration >> calibration_data[cg_y][cg_x].coordinates.x >> calibration_data[cg_y][cg_x].coordinates.y)) {
                    cantParseFile(monitor_calibration_file);
                    return -1;
                }

                auto& lut = calibration_data[cg_y][cg_x].data;

                int index = 0, last_index = -1;
                int red, green, blue;
                for (int i = 0; i < 256; i++) {
                    if (monitor_calibration >> index >> red >> green >> blue) {
                        // Perform some input data validation
                        if ((index != last_index + 1) || (red < 0 || red > 255) || (green < 0 || green > 255) || (blue < 0 || blue > 255)) {
                            cantParseFile(monitor_calibration_file);
                            return -1;
                        }
                        last_index = index;

                        lut[index] = { (uint16_t) red, (uint16_t) green, (uint16_t) blue};
                    } else {
                        cantParseFile(monitor_calibration_file);
                        return -1;
                    }
                }
            }
        }
    }

    // Build inverse transfer function to linearize the display+camera response
    std::array<std::array<std::array<gls::rgb_pixel_16, 256>, 3>, 3> inv_lut_grid;
    std::memset(inv_lut_grid.data(), 0, sizeof(inv_lut_grid));

    for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
        for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
            std::cout << "Computing inverse LUT at " << cg_x << ", " << cg_y << std::endl;

            const auto& lut = calibration_data[cg_y][cg_x].data;
            auto& inv_lut = inv_lut_grid[cg_y][cg_x];

            // Pretty print display+camera transfer function
            for (int i = 0; i < 256; i++) {
                std::cout << "level " << i << ":\t" << lut[i].red << ", " << lut[i].green << ", " << lut[i].blue << std::endl;
            }

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
        }
    }

    // Read image to transform

    // const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_rgb.png";

    const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_macbeth_calibration_captures/3/107976_calibration_MacbethSingle.3_rgb.png";

    const auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input);

    // Apply inverse transfer function to captured data to recover the original color values
    inputImage->apply([&inv_lut_grid, &inputImage, &calibration_data] (gls::rgb_pixel_16* p, int x, int y) {
        const bool use_interpolation = true;
        if (use_interpolation) {
            gls::point grid_coordinates = {
                std::clamp(x, inset, inputImage->width - inset - 1),
                std::clamp(y, inset, inputImage->height - inset - 1)
            };

            for (int cg_y = 0; cg_y < calibration_grid.height - 1; cg_y++) {
                for (int cg_x = 0; cg_x < calibration_grid.width - 1; cg_x++) {
                    const gls::rectangle quadrant(calibration_data[cg_y][cg_x].coordinates,
                                                  calibration_data[cg_y+1][cg_x+1].coordinates);
                    if (quadrant.contains(grid_coordinates)) {
                        float wx = (grid_coordinates.x - quadrant.x) / (float) quadrant.width;
                        float wy = (grid_coordinates.y - quadrant.y) / (float) quadrant.height;

                        const auto p00 = lookup(inv_lut_grid[cg_y][cg_x], *p);
                        const auto p10 = lookup(inv_lut_grid[cg_y][cg_x + 1], *p);
                        const auto p01 = lookup(inv_lut_grid[cg_y + 1][cg_x], *p);
                        const auto p11 = lookup(inv_lut_grid[cg_y + 1][cg_x + 1], *p);

                        *p = lerp(lerp(p00, p10, wx),
                                  lerp(p01, p11, wx), wy);
                    }
                }
            }
        } else {
            *p = lookup(inv_lut_grid[1][1], *p);
        }
    });

    // Write out transformed image

//    inputImage->write_png_file("/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_linrearized.png");

    inputImage->write_png_file("/Users/fabio/Desktop/107976_calibration_MacbethSingle.3_linrearized.png");

    return 0;
}
