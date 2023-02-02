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

//std::array<gls::rectangle, 9> monitorCordinates = {
//    {
//        { 161,   204,    1626,   720 },
//        { 2210,  215,    1561,   696 },
//        { 4176,  201,    1622,   681 },
//        { 162,   1359,   1621,   707 },
//        { 2204,  1357,   1567,   706 },
//        { 4159,  1338,   1635,   720 },
//        { 153,   2531,   1630,   701 },
//        { 2200,  2520,   1554,   700 },
//        { 4168,  2530,   1623,   713 }
//    }
//};

std::array<gls::rectangle, 9> monitorCordinates = {
    {
        { 113,  187,    1637,   716 },
        { 2160, 194,    1566,   704 },
        { 4128, 165,    1634,   701 },
        { 114,  1326,   1635,   726 },
        { 2164, 1329,   1556,   711 },
        { 4116, 1314,   1637,   722 },
        { 116,  2505,   1626,   706 },
        { 2163, 2494,   1540,   706 },
        { 4123, 2507,   1617,   713 }
    }
};

constexpr const int inset = 10;

gls::point constrain(const gls::point& p, const gls::rectangle r) {
    return { std::clamp(p.x, r.x, r.x + r.width), std::clamp(p.y, r.y, r.y + r.height) };
}

std::vector<gls::point> sampleGrid() {
    std::vector<gls::point> result;
    result.reserve(9 * monitorCordinates.size());

    for (const auto r : monitorCordinates) {
        result.push_back({ r.x + inset,             r.y + inset });
        result.push_back({ r.x + inset,             r.y + r.height / 2 });
        result.push_back({ r.x + inset,             r.y + r.height - inset });
        result.push_back({ r.x + r.width / 2,       r.y + inset });
        result.push_back({ r.x + r.width / 2,       r.y + r.height / 2 });
        result.push_back({ r.x + r.width / 2,       r.y + r.height - inset });
        result.push_back({ r.x + r.width - inset,   r.y + inset });
        result.push_back({ r.x + r.width - inset,   r.y + r.height / 2 });
        result.push_back({ r.x + r.width - inset,   r.y + r.height - inset });
    }
    return result;
}

std::pair<std::array<std::array<gls::point, 2>, 2>, std::pair<float, float>> pointToQuad(const gls::point& p) {
    for (const auto r : monitorCordinates) {
        if (r.contains(p)) {
            const auto p_inset = constrain(p, { r.x + inset, r.y + inset, r.x + r.width - inset, r.y + r.height - inset } );
            const auto p_off = p_inset - gls::point { r.x, r.y };
            const auto quadtant_size = gls::size { r.width / 3, r.height / 3 };
            const auto quadrant = gls::point { p_off.x % quadtant_size.width, p_off.y % quadtant_size.height };
            return {
                {{
                    quadrant,
                    quadrant + gls::point { quadtant_size.width, 0 },
                    quadrant + gls::point { 0, quadtant_size.height },
                    quadrant + gls::point { quadtant_size.width, quadtant_size.height }
                }},
                {
                    (p_off.x - quadrant.x) / (float) quadtant_size.width,
                    (p_off.y - quadrant.y) / (float) quadtant_size.height
                }
            };
        }
    }
    return {
        {{
            gls::point { 0, 0 },
            gls::point { 0, 0 },
            gls::point { 0, 0 },
            gls::point { 0, 0 }
        }},
        {0, 0}
    };
}

std::array<std::array<gls::point, 3>, 3> computeCalibrationCoordinates(const gls::rectangle& r) {
    return {{
        {{{ r.x + inset, r.y + inset },            { r.x + r.width / 2, r.y + inset },               { r.x + r.width - inset, r.y + inset }}},
        {{{ r.x + inset, r.y + r.height / 2 },     { r.x + r.width / 2, r.y + r.height / 2 },        { r.x + r.width - inset, r.y + r.height / 2 }}},
        {{{ r.x + inset, r.y + r.height - inset }, { r.x + r.width / 2, r.y + r.height - inset },    { r.x + r.width - inset, r.y + r.height - inset }}}
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
    typedef std::array<std::array<monitor_response, calibration_grid.width>, calibration_grid.height> monitor_calibration;
    std::array<monitor_calibration, 9> calibration_data;

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
                // 1_calibration_greyscale_247.png.3_rgb.png
                const std::regex filename_regex("\([0-9]+\)_calibration_greyscale_\([0-9]+\).png.\([0-9]\)_rgb");
                std::smatch regex_match;
                if (std::regex_match(filename, regex_match, filename_regex)) {
                    captureNumber = std::atoi(regex_match[1].str().c_str());
                    pixelLevel = std::atoi(regex_match[2].str().c_str());
                    exposureIndex = std::atoi(regex_match[3].str().c_str());
                }

                std::cout << "Processing file " << filename << ", captureNumber: " << captureNumber << ", pixelLevel: " << pixelLevel << ", exposureIndex: " << exposureIndex << std::endl;

                const auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input_path.string());

                for (int m = 0; m < 9; m++) {
                    const auto calibration_coordinates = computeCalibrationCoordinates(monitorCordinates[m]);

                    for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
                        for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                            const auto& p = calibration_data[m][cg_y][cg_x].coordinates = calibration_coordinates[cg_y][cg_x];
                            auto& lut = calibration_data[m][cg_y][cg_x].data;

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
        }

        // Make sure lookup table data is monotonic
        for (int m = 0; m < 9; m++) {
            for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
                for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                    auto& lut = calibration_data[m][cg_y][cg_x].data;
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
        }

        // Save monitor calibration data to file
        std::fstream monitor_calibration { monitor_calibration_file, monitor_calibration.out };

        // monitor_calibration << monitor_grid.width << " " << monitor_grid.height << std::endl;
        for (int m = 0; m < 9; m++) {
            monitor_calibration << m << std::endl;

            // monitor_calibration << calibration_grid.width << " " << calibration_grid.height << std::endl;
            for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
                for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                    monitor_calibration << calibration_data[m][cg_y][cg_x].coordinates.x << " " << calibration_data[m][cg_y][cg_x].coordinates.y << std::endl;

                    const auto& lut = calibration_data[m][cg_y][cg_x].data;
                    for (int i = 0; i < 256; i++) {
                        monitor_calibration << i << " " << lut[i].red << " " << lut[i].green << " " << lut[i].blue << std::endl;
                    }
                }
            }
        }
    } else {
        std::ifstream monitor_calibration { monitor_calibration_file, std::ios::in };

        for (int m = 0; m < 9; m++) {
            int monitor;
            monitor_calibration >> monitor;
            if (monitor != m ) {
                cantParseFile(monitor_calibration_file);
                return -1;
            }

            for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
                for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                    std::cout << "Reading grid element " << cg_x << ", " << cg_y << std::endl;

                    if (!(monitor_calibration >> calibration_data[m][cg_y][cg_x].coordinates.x >> calibration_data[m][cg_y][cg_x].coordinates.y)) {
                        cantParseFile(monitor_calibration_file);
                        return -1;
                    }

                    auto& lut = calibration_data[m][cg_y][cg_x].data;

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
    }

    // Build inverse transfer function to linearize the display+camera response
    std::array<std::array<std::array<std::array<gls::rgb_pixel_16, 256>, 3>, 3>, 9> inv_lut_grid;
    std::memset(inv_lut_grid.data(), 0, sizeof(inv_lut_grid));

    for (int m = 0; m < 9; m++) {
        for (int cg_y = 0; cg_y < calibration_grid.height; cg_y++) {
            for (int cg_x = 0; cg_x < calibration_grid.width; cg_x++) {
                std::cout << "Computing inverse LUT at " << cg_x << ", " << cg_y << std::endl;

                const auto& lut = calibration_data[m][cg_y][cg_x].data;
                auto& inv_lut = inv_lut_grid[m][cg_y][cg_x];

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
    }

    // Read image to transform

    // const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_rgb.png";

    // const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_macbeth_calibration_captures/3/107976_calibration_MacbethSingle.3_rgb.png";
    const std::string input = "/Users/fabio/work/2023-01-30_Calibration/rgb_data/3/1_calibration_MacbethSingle.png.3_rgb.png";

    auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input);

    // Efficiently apply inverse transfer function to captured data to recover the original color values
    for (int m = 0; m < 9; m++) {
        // Iterate over monitors
        for (int cg_y = 0; cg_y < calibration_grid.height - 1; cg_y++) {
            for (int cg_x = 0; cg_x < calibration_grid.width - 1; cg_x++) {
                // Iterate over calibration grid coordinates
                const gls::rectangle quadrant(calibration_data[m][cg_y][cg_x].coordinates,
                                              calibration_data[m][cg_y+1][cg_x+1].coordinates);

                // LUTs at the corners of this quad
                const auto& lut00 = inv_lut_grid[m][cg_y][cg_x];
                const auto& lut01 = inv_lut_grid[m][cg_y][cg_x + 1];
                const auto& lut10 = inv_lut_grid[m][cg_y + 1][cg_x];
                const auto& lut11 = inv_lut_grid[m][cg_y + 1][cg_x + 1];

                // Apply inverse LUT to the sub-image identified by this quadrant
                gls::image<gls::rgb_pixel_16>(inputImage.get(), quadrant).apply([&quadrant, &lut00, &lut01, &lut10, &lut11](gls::rgb_pixel_16* p, int x, int y) {
                    float wx = x / (float) quadrant.width;
                    float wy = y / (float) quadrant.height;

                    const auto& p00 = lookup(lut00, *p);
                    const auto& p10 = lookup(lut01, *p);
                    const auto& p01 = lookup(lut10, *p);
                    const auto& p11 = lookup(lut11, *p);

                    *p = lerp(lerp(p00, p10, wx),
                              lerp(p01, p11, wx), wy);
                });
            }
        }
    }

    // Write out transformed image

//    inputImage->write_png_file("/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_linrearized.png");

    inputImage->write_png_file("/Users/fabio/Desktop/1_calibration_MacbethSingle.3_linrearized.png");

    return 0;
}
