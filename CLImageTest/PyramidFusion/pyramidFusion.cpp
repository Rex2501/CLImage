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

#include <iostream>
#include <filesystem>
#include <string>
#include <cmath>
#include <chrono>
#include <memory>
#include <ranges>
#include <set>

#include "gls_logging.h"
#include "gls_image.hpp"
#include "gls_linalg.hpp"

#include "raw_converter.hpp"
#include "CameraCalibration.hpp"

#include "SURF.hpp"
#include "RANSAC.hpp"

static const char* TAG = "PyramidFusion";

std::vector<std::filesystem::path> parseDirectory(const std::string& dir) {
    std::set<std::filesystem::path> directory_listing;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".dng" || entry.path().extension() == ".DNG")) {
            directory_listing.insert(entry.path());
        }
    }
    return std::vector<std::filesystem::path>(directory_listing.begin(), directory_listing.end());
}

template <typename T>
gls::image<float> asGrayscaleFloat(const gls::image<T>& image) {
    gls::image<float> grayscale(image.width, image.height);

    // Convert to grayscale and normalize values in [0..1]
    grayscale.apply([&](float *p, int x, int y) {
        const auto& pIn = image[y][x];
        *p = std::clamp((pIn.red * 0.299 + pIn.green * 0.587 + pIn.blue * 0.114) / 255.0, 0.0, 1.0);
    });

    return grayscale;
}

gls::image<gls::rgb_pixel>::unique_ptr runPipeline(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    // return demosaicIMX571DNG(rawConverter, input_path);
    // return demosaicSonya6400DNG(rawConverter, input_path);
    // return demosaicCanonEOSRPDNG(rawConverter, input_path);
    return demosaiciPhone11(rawConverter, input_path);
    // return demosaicRicohGRIII2DNG(rawConverter, input_path);
    // return demosaicLeicaQ2DNG(rawConverter, input_path);
}

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        std::cout << "Please provide a directory path..." << std::endl;
    }

    std::vector<std::filesystem::path> input_files = parseDirectory(argv[1]);
    if (input_files.size() < 2) {
        std::cout << "we need at least two files..." << std::endl;
    }

    const auto& reference_image_path = input_files[0];
    LOG_INFO(TAG) << "Reference Image: " << reference_image_path.filename() << std::endl;

    gls::OpenCLContext glsContext("");

    RawConverter rawConverter(&glsContext);

    const auto reference_image_rgb = runPipeline(&rawConverter, reference_image_path.string());

    // const auto reference_image_rgb = gls::image<gls::rgb_pixel>::read_png_file(reference_image_path.string());

    const auto reference_image = asGrayscaleFloat(*reference_image_rgb);

    auto surf = gls::SURF::makeInstance(&glsContext, reference_image.width, reference_image.height,
                                        /*max_features=*/ 1500, /*nOctaves=*/ 4, /*nOctaveLayers=*/ 2, /*hessianThreshold=*/ 0.02);

    auto reference_keypoints = std::make_unique<std::vector<KeyPoint>>();
    gls::image<float>::unique_ptr reference_descriptors;
    surf->detectAndCompute(reference_image, reference_keypoints.get(), &reference_descriptors);

    gls::image<gls::rgb_pixel> result_image(reference_image_rgb->size());
    std::copy(reference_image_rgb->pixels().begin(), reference_image_rgb->pixels().end(), result_image.pixels().begin());

    int fused_images = 1;
    for (const auto& image_path : std::span(&input_files[1], &input_files[input_files.size()])) {
        LOG_INFO(TAG) << "Processing: " << image_path.filename() << std::endl;

        const auto image_rgb = runPipeline(&rawConverter, image_path.string());
        // const auto image_rgb = gls::image<gls::rgba_pixel>::read_png_file(image_path.string());
        const auto image = asGrayscaleFloat(*image_rgb);

        auto image_keypoints = std::make_unique<std::vector<KeyPoint>>();
        gls::image<float>::unique_ptr image_descriptors;
        surf->detectAndCompute(image, image_keypoints.get(), &image_descriptors);

        std::vector<gls::DMatch> matchedPoints = surf->matchKeyPoints(*reference_descriptors, *image_descriptors);

        // Limit the max number of matches
        int max_matches = std::min(300, (int) matchedPoints.size());

        // Convert to Point2D format
        std::vector<std::pair<Point2f, Point2f>> matchpoints(max_matches);
        std::transform(&matchedPoints[0], &matchedPoints[max_matches], matchpoints.begin(),
                       [&reference_keypoints, &image_keypoints](const auto &mp) {
            return std::pair {
                (*reference_keypoints)[mp.queryIdx].pt,
                (*image_keypoints)[mp.trainIdx].pt
            };
        });

        std::vector<int> inliers;
        const auto homography = gls::RANSAC(matchpoints, /*threshold=*/ 1, /*max_iterations=*/ 2000, &inliers);

        std::cout << "Homography:\n" << homography << std::endl;

        auto cl_image = gls::cl_image_2d<gls::rgba_pixel>(glsContext.clContext(), image_rgb->size());
        auto cl_image_cpu = cl_image.mapImage(CL_MAP_WRITE);
        cl_image_cpu.apply([&image_rgb](gls::rgba_pixel *p, int x, int y) {
            const auto& pin = (*image_rgb)[y][x];
            *p = { pin.red, pin.green, pin.blue, 255 };
        });
        cl_image.unmapImage(cl_image_cpu);

        gls::cl_image_2d<gls::rgba_pixel> registered_image(glsContext.clContext(), image.width, image.height);
        gls::clRegisterImage(&glsContext, cl_image, &registered_image, homography);

        auto registered_image_cpu = registered_image.mapImage(CL_MAP_READ);
        registered_image_cpu.apply([&result_image, fused_images](gls::rgba_pixel *p, int x, int y) {
            const auto& pin = result_image[y][x];
            result_image[y][x] = {
                (uint8_t) ((fused_images * pin.red + p->red) / (fused_images + 1)),
                (uint8_t) ((fused_images * pin.green + p->green) / (fused_images + 1)),
                (uint8_t) ((fused_images * pin.blue + p->blue) / (fused_images + 1))
            };
        });
        registered_image.unmapImage(registered_image_cpu);
        fused_images++;
    }

    result_image.write_png_file(reference_image_path.parent_path() / "fused.png");

    return 0;
}
