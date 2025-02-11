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

#include "gls_logging.h"
#include "gls_image.hpp"
#include "gls_linalg.hpp"

#include "SURF.hpp"
#include "RANSAC.hpp"

#if __APPLE__
#define PATH "/Users/fabio/work/ImageRegistration/RAW/F8/"
#else
#define PATH "/data/local/tmp/"
#endif

void testSURF() {
    auto glsContext = new gls::OpenCLContext("");

    auto srcImg1_ = gls::image<gls::rgb_pixel>::read_png_file(PATH "L1008535.png");
    auto srcImg2_ = gls::image<gls::rgb_pixel>::read_png_file(PATH "L1008536.png");

//    const auto srcImg1_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg1_full, gls::rectangle {0, srcImg1_full->height/2, srcImg1_full->width, srcImg1_full->height/2});
//    const auto srcImg2_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg2_full, gls::rectangle {0, srcImg2_full->height/2, srcImg2_full->width, srcImg2_full->height/2});

    gls::image<float> srcImg1(srcImg1_->width, srcImg1_->height);
    gls::image<float> srcImg2(srcImg2_->width, srcImg2_->height);

    std::cout << "image size: " << srcImg1_->width << "x" << srcImg1_->height << std::endl;

    // Convert to grayscale and normalize values in [0..1]
    srcImg1.apply([&](float *p, int x, int y) {
        const gls::rgb_pixel& pIn = (*srcImg1_)[y][x];
        *p = std::clamp((pIn.red * 0.299 + pIn.green * 0.587 + pIn.blue * 0.114) / 255.0, 0.0, 1.0);
    });

    srcImg2.apply([&](float *p, int x, int y) {
        const gls::rgb_pixel& pIn = (*srcImg2_)[y][x];
        *p = std::clamp((pIn.red * 0.299 + pIn.green * 0.587 + pIn.blue * 0.114) / 255.0, 0.0, 1.0);
    });

    auto t_start = std::chrono::high_resolution_clock::now();

    auto matchpoints = gls::SURF::detection(glsContext, srcImg1, srcImg2);

    auto t_surf = std::chrono::high_resolution_clock::now();

    printf("Feature Dection matched %d features\n", (int) matchpoints.size());

    // Limit the max number of matches
    int max_matches = 300;
    if (matchpoints.size() > max_matches) {
        printf("SURF_Detection - dropping: %d features out of %d\n", (int) matchpoints.size() - max_matches, (int) matchpoints.size());
        matchpoints.erase(matchpoints.begin() + max_matches, matchpoints.end());
    }

    std::vector<int> inliers;
    const auto homography = gls::RANSAC(matchpoints, /*threshold=*/ 1, /*max_iterations=*/ 2000, &inliers);

    std::cout << "Homography:\n" << homography << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();

    double surf_time_ms = std::chrono::duration<double, std::milli>(t_surf-t_start).count();
    double ransac_time_ms = std::chrono::duration<double, std::milli>(t_end-t_surf).count();
    double total_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    printf("SURF Time: %f, RANSAC Time: %f, Total Execution Time: %f\n", surf_time_ms, ransac_time_ms, total_time_ms);

    auto inputImage1 = gls::cl_image_2d<gls::rgba_pixel>(glsContext->clContext(), srcImg1_->width, srcImg1_->height);
    auto inputImage1Cpu = inputImage1.mapImage(CL_MAP_WRITE);
    inputImage1Cpu.apply([&srcImg1_](gls::rgba_pixel* p, int x, int y) {
        const auto& pIn = (*srcImg1_)[y][x];
        *p = { pIn.red, pIn.green, pIn.blue, 255 };
    });
    inputImage1.unmapImage(inputImage1Cpu);

    auto inputImage2 = gls::cl_image_2d<gls::rgba_pixel>(glsContext->clContext(), srcImg2_->width, srcImg2_->height);
    auto inputImage2Cpu = inputImage2.mapImage(CL_MAP_WRITE);
    inputImage2Cpu.apply([&srcImg2_](gls::rgba_pixel* p, int x, int y) {
        const auto& pIn = (*srcImg2_)[y][x];
        *p = { pIn.red, pIn.green, pIn.blue, 255 };
    });
    inputImage2.unmapImage(inputImage2Cpu);

    auto outputImage = gls::cl_image_2d<gls::rgba_pixel>(glsContext->clContext(), srcImg1_->width, srcImg1_->height);
    {
        gls::clRegisterAndFuse(glsContext, inputImage1, inputImage2, &outputImage, homography);

        auto outputImageCpu = outputImage.mapImage(CL_MAP_READ);

        for (const auto& mp : matchpoints) {
            outputImageCpu[(int) mp.first.y][(int) mp.first.x] = {255, 0, 0, 255};
        }
        for (const auto& i : inliers) {
            const auto& mp = matchpoints[i];
            outputImageCpu[(int) mp.first.y][(int) mp.first.x] = {0, 255, 0, 255};
        }

        outputImageCpu.write_png_file(PATH "fused.png", /*skip_alpha=*/ true);
        outputImage.unmapImage(outputImageCpu);
    }
    {
        gls::clRegisterImage(glsContext, inputImage2, &outputImage, homography);

        auto outputImageCpu = outputImage.mapImage(CL_MAP_READ | CL_MAP_WRITE);
        outputImageCpu.write_png_file(PATH "registered.png", /*skip_alpha=*/ true);
        outputImage.unmapImage(outputImageCpu);
    }
    {
        for (const auto& mp : matchpoints) {
            (*srcImg1_)[(int) mp.first.y][(int) mp.first.x] = {255, 0, 0};
        }
        for (const auto& i : inliers) {
            const auto& mp = matchpoints[i];
            (*srcImg1_)[(int) mp.first.y][(int) mp.first.x] = {0, 255, 0};
        }
        srcImg1_->write_png_file(PATH "srcImg1MatchPoints.png", /*skip_alpha=*/ true);
    }
    {
        for (const auto& mp : matchpoints) {
            (*srcImg2_)[(int) mp.second.y][(int) mp.second.x] = {255, 0, 0};
        }
        for (const auto& i : inliers) {
            const auto& mp = matchpoints[i];
            (*srcImg2_)[(int) mp.second.y][(int) mp.second.x] = {0, 255, 0};
        }
        srcImg2_->write_png_file(PATH "srcImg2MatchPoints.png", /*skip_alpha=*/ true);
    }}

int main(int argc, const char * argv[]) {
    std::cout << "ImageRegistration Tests!\n";

//    try {
        testSURF();
//    } catch (const cl::Error& e) {
//        std::cout << "cl::Error " << e.what() << " - " << gls::clStatusToString(e.err()) << std::endl;
//    }

    return 0;
}
