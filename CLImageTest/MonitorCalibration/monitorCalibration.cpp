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

#include "gls_image.hpp"

int main(int argc, const char * argv[]) {
    auto input_path = std::filesystem::path(argv[1]);

    std::cout << "Processing Directory: " << input_path.filename() << std::endl;

    auto input_dir = std::filesystem::directory_entry(input_path).is_directory() ? input_path : input_path.parent_path();
    std::vector<std::filesystem::path> directory_listing;
    std::copy(std::filesystem::directory_iterator(input_dir), std::filesystem::directory_iterator(),
              std::back_inserter(directory_listing));
    std::sort(directory_listing.begin(), directory_listing.end());

    gls::rgb_pixel_16 lut[256];

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

            // lut[pixelLevel] = (*inputImage)[inputImage->height/2][inputImage->width/2];

            std::cout << "Processing file " << filename << ", captureNumber: " << captureNumber << ", pixelLevel: " << pixelLevel << ", exposureIndex: " << exposureIndex << std::endl;
        }
    }

    gls::rgb_pixel_16 last_value = lut[255];
    for (int i = 254; i >= 0; i--) {
        if (lut[i].red > last_value.red) {
            lut[i].red = last_value.red;
        } else {
            last_value.red = lut[i].red;
        }
        if (lut[i].green > last_value.green) {
            lut[i].green = last_value.green;
        } else {
            last_value.green = lut[i].green;
        }
        if (lut[i].blue > last_value.blue) {
            lut[i].blue = last_value.blue;
        } else {
            last_value.blue = lut[i].blue;
        }
    }

    for (int i = 0; i < 256; i++) {
        std::cout << "level " << i << ":\t" << lut[i].red << ", " << lut[i].green << ", " << lut[i].blue << std::endl;
    }

    gls::rgb_pixel_16 inv_lut[256];
    std::memset(inv_lut, 0, sizeof(inv_lut));

    for (int i = 0; i < 256; i++) {
        const auto entry = lut[i];
        inv_lut[entry.red].red = i;
        inv_lut[entry.green].green = i;
        inv_lut[entry.blue].blue = i;
    }

    int last_red_idx = 0;
    int last_green_idx = 0;
    int last_blue_idx = 0;
    int last_red_val = 0;
    int last_green_val = 0;
    int last_blue_val = 0;
    for (int i = 0; i < 256; i++) {
        auto entry = inv_lut[i];
        if (i == 255) {
            if (entry.red == 0) {
                entry.red = 255;
            }
            if (entry.green == 0) {
                entry.green = 255;
            }
            if (entry.blue == 0) {
                entry.blue = 255;
            }
        }
        if (entry.red != 0) {
            if (i - last_red_idx > 1) {
                float delta = (entry.red - last_red_val) / (float) (i - last_red_idx);
                for (int j = 0; j < i - last_red_idx; j++) {
                    inv_lut[last_red_idx + j].red = last_red_val + delta * j;
                }
            }
            last_red_idx = i;
            inv_lut[i].red = last_red_val = entry.red;
        }
        if (entry.green != 0) {
            if (i - last_green_idx > 1) {
                float delta = (entry.green - last_green_val) / (float) (i - last_green_idx);
                for (int j = 0; j < i - last_green_idx; j++) {
                    inv_lut[last_green_idx + j].green = last_green_val + delta * j;
                }
            }
            last_green_idx = i;
            last_green_val = entry.green;
            inv_lut[i].green = last_green_val = entry.green;
        }
        if (entry.blue != 0) {
            if (i - last_blue_idx > 1) {
                float delta = (entry.blue - last_blue_val) / (float) (i - last_blue_idx);
                for (int j = 0; j < i - last_blue_idx; j++) {
                    inv_lut[last_blue_idx + j].blue = last_blue_val + delta * j;
                }
            }
            last_blue_idx = i;
            last_blue_val = entry.blue;
            inv_lut[i].blue = last_blue_val = entry.blue;
        }
    }

//    for (int i = 255; i >= 0; i--) {
//        inv_lut[i].red -= (int) (inv_lut[0].red * (255 - i) / 255.0f);
//        inv_lut[i].green -= (int) (inv_lut[0].green * (255 - i) / 255.0f);
//        inv_lut[i].blue -= (int) (inv_lut[0].blue * (255 - i) / 255.0f);
//    }

    for (int i = 0; i < 256; i++) {
        std::cout << "inv level " << i << ":\t" << inv_lut[i].red << ", " << inv_lut[i].green << ", " << inv_lut[i].blue << std::endl;
    }

    // const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_rgb.png";

    const std::string input = "/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_macbeth_calibration_captures/3/107976_calibration_MacbethSingle.3_rgb.png";

    const auto inputImage = gls::image<gls::rgb_pixel_16>::read_png_file(input);

    inputImage->apply([&inv_lut] (gls::rgb_pixel_16* p, int x, int y) {
        const uint8_t red8 = p->red >> 8;
        const uint8_t green8 = p->green >> 8;
        const uint8_t blue8 = p->blue >> 8;

        const float red_ratio = inv_lut[red8].red / (p->red / 256.0f);
        const float green_ratio = inv_lut[green8].green / (p->green / 256.0f);
        const float blue_ratio = inv_lut[blue8].blue / (p->blue / 256.0f);

        *p = {
            (uint16_t) (red8 > 0 ? std::clamp(p->red * red_ratio, 0.0f, (float) 0xffff) : inv_lut[red8].red << 8),
            (uint16_t) (green8 > 0 ? std::clamp(p->green * green_ratio, 0.0f, (float) 0xffff) : inv_lut[green8].green << 8),
            (uint16_t) (blue8 > 0 ? std::clamp(p->blue * blue_ratio, 0.0f, (float) 0xffff) : inv_lut[blue8].blue << 8)
        };
    });

//    inputImage->write_png_file("/Users/fabio/work/2023-01-13_Calibration/rgb_data/2023-01-13_greyscale_calibration_captures/3/107976_calibration_greyscale_128.3_linrearized.png");

    inputImage->write_png_file("/Users/fabio/Desktop/107976_calibration_MacbethSingle.3_linrearized.png");

    return 0;
}
