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

gls::cl_image_2d<gls::rgba_pixel_float>* runPipeline(gls::OpenCLContext* glsContext, RawConverter* rawConverter, const std::filesystem::path& input_path, std::unique_ptr<DemosaicParameters>* demosaicParameters) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    if (*demosaicParameters == nullptr) {
        const auto cameraCalibration = getLeicaQ2Calibration(); // getIPhone11Calibration();
        *demosaicParameters = cameraCalibration->getDemosaicParameters(*inputImage, &dng_metadata, &exif_metadata);
    }

    const auto demosaicedImage = rawConverter->demosaic(*inputImage, demosaicParameters->get(), /*calibrateFromImage=*/ true);

    return demosaicedImage;
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

    std::unique_ptr<DemosaicParameters> demosaicParameters = nullptr;
    auto reference_image_rgb = runPipeline(&glsContext, &rawConverter, reference_image_path.string(), &demosaicParameters);

    gls::cl_image_2d<float> cl_reference_image(glsContext.clContext(), reference_image_rgb->width, reference_image_rgb->height);
    convertToGrayscale(&glsContext, *reference_image_rgb, &cl_reference_image, *demosaicParameters);

    const auto reference_image = cl_reference_image.mapImage();

    auto surf = gls::SURF::makeInstance(&glsContext, reference_image.width, reference_image.height,
                                        /*max_features=*/ 1500, /*nOctaves=*/ 4, /*nOctaveLayers=*/ 2, /*hessianThreshold=*/ 0.02);

    auto reference_keypoints = std::make_unique<std::vector<KeyPoint>>();
    gls::image<float>::unique_ptr reference_descriptors;
    surf->detectAndCompute(reference_image, reference_keypoints.get(), &reference_descriptors);

    cl_reference_image.unmapImage(reference_image);

    // Convert linear image to YCbCr for fusion
    const auto cam_to_ycbcr = cam_ycbcr(demosaicParameters->rgb_cam);
    transformImage(&glsContext, *reference_image_rgb, reference_image_rgb, cam_to_ycbcr);

    // The identity homography is not really used here as this is the first image
    rawConverter.fuseFrame(*reference_image_rgb, gls::Matrix<3, 3>::identity(),
                           demosaicParameters.get(), /*calibrateFromImage=*/ false);

    int fused_images = 1;
    for (const auto& image_path : std::span(&input_files[1], &input_files[input_files.size()])) {
        LOG_INFO(TAG) << "Processing: " << image_path.filename() << std::endl;

        const auto image_rgb = runPipeline(&glsContext, &rawConverter, image_path.string(), &demosaicParameters);

        gls::cl_image_2d<float> cl_image(glsContext.clContext(), image_rgb->width, image_rgb->height);
        convertToGrayscale(&glsContext, *image_rgb, &cl_image, *demosaicParameters);

        auto image_keypoints = std::make_unique<std::vector<KeyPoint>>();
        gls::image<float>::unique_ptr image_descriptors;

        const auto image = cl_image.mapImage();

        surf->detectAndCompute(image, image_keypoints.get(), &image_descriptors);

        cl_image.unmapImage(image);

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

        // Convert to YCbCr for fusion
        transformImage(&glsContext, *image_rgb, image_rgb, cam_to_ycbcr);

        // Fuse the image with the rest applying the homography we just found
        rawConverter.fuseFrame(*image_rgb, homography, demosaicParameters.get(), /*calibrateFromImage=*/ false);

        fused_images++;
    }

    // Save result
    {
        const auto cl_result_image = rawConverter.getFusedImage();
        const auto denoisedImage = rawConverter.denoise(*cl_result_image, demosaicParameters.get(), /*calibrateFromImage=*/ true);

        // Convert result back to camera RGB
        const auto normalized_ycbcr_to_cam = inverse(cam_to_ycbcr) * demosaicParameters->exposure_multiplier;
        transformImage(&glsContext, *denoisedImage, denoisedImage, normalized_ycbcr_to_cam);

        const auto sRGBImage = rawConverter.postProcess(*denoisedImage, *demosaicParameters);
        const auto result_image = RawConverter::convertToRGBImage(*sRGBImage);

        result_image->write_png_file(reference_image_path.parent_path() / "fused_NTB.png");
    }

    return 0;
}
