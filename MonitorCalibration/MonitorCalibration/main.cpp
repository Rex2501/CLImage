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

struct MonitorResponse {
    gls::point location = { -1, -1 };
    std::array<gls::rgb_pixel_16, 256> response_lut;
    std::array<gls::rgb_pixel_16, 256> inverse_response_lut;

    void sampleImageAt(const gls::image<gls::rgb_pixel_16>& inputImage, const gls::point& p, const int pixelLevel) {
        if (location != gls::point { -1, -1 }) {
            assert(location == p);
        } else {
            location = p;
        }

        gls::Vector<3> pixelValue = gls::Vector<3>::zeros();
        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                pixelValue += gls::Vector<3>(inputImage[p.y + y][p.x + x].v);
            }
        }
        pixelValue /= 25.0f;
        response_lut[pixelLevel] = pixelValue / 256.0f;
    }

    void ensureMonotonic() {
        // Make sure lookup table data is monotonic
        gls::rgb_pixel_16 last_value = response_lut[255];
        for (int i = 254; i >= 0; i--) {
            for (int c = 0; c < 3; c++) {
                if (response_lut[i][c] > last_value[c]) {
                    response_lut[i][c] = last_value[c];
                } else {
                    last_value[c] = response_lut[i][c];
                }
            }
        }
    }

    gls::rgb_pixel_16 lookup(const gls::rgb_pixel_16& val) const {
        gls::rgb_pixel_16 result;
        for (int c = 0; c < 3; c++) {
            const uint8_t val8 = val[c] >> 8;

            if (val8 > 0) {
                // For input values greater than the minimum LUT value, interpolate the 8 bit LUT entry over 16 bit data
                const float ratio = inverse_response_lut[val8][c] / (val[c] / 256.0f);
                result[c] = std::clamp(val[c] * ratio, 0.0f, (float) 0xffff);
            } else {
                // Oterwise just use the value from the LUT
                result[c] = inverse_response_lut[val8][c] << 8;
            }
        }
        return result;
    }

    MonitorResponse() {
        std::memset(response_lut.data(), 0, sizeof(response_lut));
        std::memset(inverse_response_lut.data(), 0, sizeof(inverse_response_lut));
    }
};

//// Glass Camera Last capture
//const constexpr int monitors = 9;
//static const std::array<gls::rectangle, monitors> monitorCordinates = {
//    {
//        { 113,  187,    1637,   716 },
//        { 2160, 194,    1566,   704 },
//        { 4128, 165,    1634,   701 },
//        { 114,  1326,   1635,   726 },
//        { 2164, 1329,   1556,   711 },
//        { 4116, 1314,   1637,   722 },
//        { 116,  2505,   1626,   706 },
//        { 2163, 2494,   1540,   706 },
//        { 4123, 2507,   1617,   713 }
//    }
//};

// Asus Calibration
const constexpr int monitors = 9;
static const std::array<gls::rectangle, monitors> monitorCordinates = {
    {
        { 2427, 1936, 951, 534 },
        { 3624, 1945, 948, 531 },
        { 4807, 1938, 948, 534 },
        { 2433, 2781, 945, 525 },
        { 3624, 2789, 935, 517 },
        { 4795, 2785, 943, 525 },
        { 2441, 3623, 933, 516 },
        { 3619, 3626, 928, 518 },
        { 4792, 3642, 927, 518 }
    }
};

template <int GridWidth, int GridHeight, int MonitorsCount>
struct MonitorCalibration : public std::array<std::array<std::array<MonitorResponse, GridWidth>, GridHeight>, MonitorsCount> {
    static std::array<std::array<gls::point, 3>, 3> computeCalibrationCoordinates(const gls::rectangle& r) {
        constexpr const int inset = 10;
        return {{
            {{{ r.x + inset, r.y + inset },            { r.x + r.width / 2, r.y + inset },               { r.x + r.width - inset, r.y + inset }}},
            {{{ r.x + inset, r.y + r.height / 2 },     { r.x + r.width / 2, r.y + r.height / 2 },        { r.x + r.width - inset, r.y + r.height / 2 }}},
            {{{ r.x + inset, r.y + r.height - inset }, { r.x + r.width / 2, r.y + r.height - inset },    { r.x + r.width - inset, r.y + r.height - inset }}}
        }};
    }

    void sampleImage(const gls::image<gls::rgb_pixel_16>& inputImage, std::array<gls::rectangle, MonitorsCount> monitorCordinates, int pixelLevel) {
        for (int m = 0; m < MonitorsCount; m++) {
            const auto calibration_coordinates = computeCalibrationCoordinates(monitorCordinates[m]);

            for (int cg_y = 0; cg_y < GridHeight; cg_y++) {
                for (int cg_x = 0; cg_x < GridWidth; cg_x++) {
                    (*this)[m][cg_y][cg_x].sampleImageAt(inputImage, calibration_coordinates[cg_y][cg_x], pixelLevel);
                }
            }
        }
    }

    void applyCalibration(gls::image<gls::rgb_pixel_16>* inputImage) const {
        for (auto monitor_calibration : *this ) {
            for (int cg_y = 0; cg_y < GridHeight - 1; cg_y++) {
                for (int cg_x = 0; cg_x < GridWidth - 1; cg_x++) {
                    // Iterate over calibration grid coordinates
                    const gls::rectangle quadrant(monitor_calibration[cg_y][cg_x].location,
                                                  monitor_calibration[cg_y+1][cg_x+1].location);

                    // LUTs at the corners of this quad
                    const auto& lut00 = monitor_calibration[cg_y][cg_x];
                    const auto& lut01 = monitor_calibration[cg_y][cg_x + 1];
                    const auto& lut10 = monitor_calibration[cg_y + 1][cg_x];
                    const auto& lut11 = monitor_calibration[cg_y + 1][cg_x + 1];

                    // Apply inverse LUT to the sub-image identified by this quadrant
                    gls::image<gls::rgb_pixel_16>(inputImage, quadrant).apply([&quadrant, &lut00, &lut01, &lut10, &lut11](gls::rgb_pixel_16* p, int x, int y) {
                        float wx = x / (float) quadrant.width;
                        float wy = y / (float) quadrant.height;

                        const auto& p00 = lut00.lookup(*p);
                        const auto& p10 = lut01.lookup(*p);
                        const auto& p01 = lut10.lookup(*p);
                        const auto& p11 = lut11.lookup(*p);

                        *p = lerp(lerp(p00, p10, wx),
                                  lerp(p01, p11, wx), wy);
                    });
                }
            }
        }
    }

    void ensureMonotonic() {
        // Make sure lookup table data is monotonic
        for (auto& monitor_calibration : *this) {
            for (auto& calibration_row : monitor_calibration) {
                for (auto& calibration_entry : calibration_row) {
                    calibration_entry.ensureMonotonic();
                }
            }
        }
    }

    void buildInverseLUT() {
        // Build inverse transfer function to linearize the display+camera response
        for (auto& monitor_calibration : *this) {
            for (auto& calibration_row : monitor_calibration) {
                for (auto& calibration_entry : calibration_row) {
                    const auto& lut = calibration_entry.response_lut;
                    auto& inv_lut = calibration_entry.inverse_response_lut;

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
    }

    void saveToFile(const std::string& monitor_calibration_file) {
        std::fstream monitor_calibration_stream { monitor_calibration_file, std::fstream::out };

        for (const auto& monitor_calibration : *this) {
            for (const auto& calibration_row : monitor_calibration) {
                for (const auto& calibration_entry : calibration_row) {
                    monitor_calibration_stream << calibration_entry.location.x << " " << calibration_entry.location.y << std::endl;

                    const auto& lut = calibration_entry.response_lut;
                    for (int i = 0; i < 256; i++) {
                        monitor_calibration_stream << i << " " << lut[i].red << " " << lut[i].green << " " << lut[i].blue << std::endl;
                    }
                }
            }
        }
    }

    void loadFromFile(const std::string& monitor_calibration_file) {
        std::ifstream monitor_calibration_stream { monitor_calibration_file, std::ios::in };

        for (auto& monitor_calibration : *this) {
            for (auto& calibration_row : monitor_calibration) {
                for (auto& calibration_entry : calibration_row) {
                    // std::cout << "Reading grid element " << cg_x << ", " << cg_y << std::endl;

                    if (!(monitor_calibration_stream >> calibration_entry.location.x >> calibration_entry.location.y)) {
                        throw std::runtime_error("Couldn't read calibration file: " + monitor_calibration_file);
                    }

                    auto& lut = calibration_entry.response_lut;

                    int index = 0, last_index = -1;
                    int red, green, blue;
                    for (int i = 0; i < 256; i++) {
                        if (monitor_calibration_stream >> index >> red >> green >> blue) {
                            // Perform some input data validation
                            if ((index != last_index + 1) || (red < 0 || red > 255) || (green < 0 || green > 255) || (blue < 0 || blue > 255)) {
                                throw std::runtime_error("Couldn't read calibration file: " + monitor_calibration_file);
                            }
                            last_index = index;

                            lut[index] = { (uint16_t) red, (uint16_t) green, (uint16_t) blue};
                        } else {
                            throw std::runtime_error("Couldn't read calibration file: " + monitor_calibration_file);
                        }
                    }
                }
            }
        }
    }
};

int main(int argc, const char * argv[]) {
    auto input_path = std::filesystem::path(argv[1]);

    std::cout << "Processing Directory: " << input_path.filename() << std::endl;

    const auto input_dir = std::filesystem::directory_entry(input_path).is_directory() ? input_path : input_path.parent_path();

    const auto monitor_calibration_file = input_dir / "MonitorCalibration.dat";

    MonitorCalibration<3, 3, 9> calibration_data;

    if (!exists(status(monitor_calibration_file))) {
        // If we don't have a calibration file, go through the calibration images and create one

        std::vector<std::filesystem::path> directory_listing;
        std::copy(std::filesystem::directory_iterator(input_dir), std::filesystem::directory_iterator(),
                  std::back_inserter(directory_listing));
        std::sort(directory_listing.begin(), directory_listing.end());

        for (const auto& input_path : directory_listing) {
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

                calibration_data.sampleImage(*inputImage, monitorCordinates, pixelLevel);
            }
        }
        calibration_data.ensureMonotonic();

        calibration_data.saveToFile(monitor_calibration_file);
    } else {
        calibration_data.loadFromFile(monitor_calibration_file);
    }

    calibration_data.buildInverseLUT();

    // Read image to transform

    // const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_rgb.png";
    // const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_macbeth_calibration_captures/3/107976_calibration_MacbethSingle.3_rgb.png";
    // const std::string input = "/Users/fabio/work/2023-01-30_Calibration/rgb_data/3/1_calibration_MacbethSingle.png.3_rgb.png";
    const std::string input = "/Users/fabio/work/asus_calibration_data/20230202_DF2KLAION_ASUS_Standard_tiled/rgb_data/5/11_calibration_MacbethSingle.png.5_rgb.png";

    auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input);

    calibration_data.applyCalibration(inputImage.get());

    // Write out transformed image

    // inputImage->write_png_file("/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_linrearized.png");
    inputImage->write_png_file("/Users/fabio/Desktop/11_calibration_MacbethSingle_linearized.5_rgb");

    return 0;
}
