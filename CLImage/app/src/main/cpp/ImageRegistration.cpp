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

#include "gls_logging.h"
#include "gls_image.hpp"

#include "gls_linalg.hpp"

#include "SURF.hpp"
#include "RANSAC.hpp"

#if __APPLE__
#define PATH "/Users/fabio/work/ImageRegistration/"
#else
#define PATH "/data/local/tmp/"
#endif

void testSURF() {
    auto glsContext = new gls::OpenCLContext("");

    const auto srcImg1_ = gls::image<gls::rgb_pixel>::read_jpeg_file(PATH "DSCF8601.jpg");
    const auto srcImg2_ = gls::image<gls::rgb_pixel>::read_jpeg_file(PATH "DSCF8603.jpg");

//    const auto srcImg1_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg1_full, gls::rectangle {0, srcImg1_full->height/2, srcImg1_full->width, srcImg1_full->height/2});
//    const auto srcImg2_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg2_full, gls::rectangle {0, srcImg2_full->height/2, srcImg2_full->width, srcImg2_full->height/2});

    gls::image<float> srcImg1(srcImg1_->width, srcImg1_->height);
    gls::image<float> srcImg2(srcImg2_->width, srcImg2_->height);

    std::cout << "image size: " << srcImg1_->width << "x" << srcImg1_->height << std::endl;

    srcImg1.apply([&](float *p, int x, int y) {
        const gls::rgb_pixel& pIn = (*srcImg1_)[y][x];
        *p = std::clamp(pIn.red * 0.299 + pIn.green * 0.587 + pIn.blue * 0.114, 0.0, 255.0);
    });

    srcImg2.apply([&](float *p, int x, int y) {
        const gls::rgb_pixel& pIn = (*srcImg2_)[y][x];
        *p = std::clamp(pIn.red * 0.299 + pIn.green * 0.587 + pIn.blue * 0.114, 0.0, 255.0);
    });

    std::vector<gls::Point2f> matchpoints1, matchpoints2;

    auto t_start = std::chrono::high_resolution_clock::now();

    bool success = gls::SURF_Detection(glsContext, srcImg1, srcImg2, &matchpoints1, &matchpoints2);
    assert(matchpoints1.size() == matchpoints2.size());
    printf("Feature Dection successful: %d, matched %d features\n", success, (int) matchpoints1.size());

    std::vector<float> transParameter;
    transParameter = gls::getRANSAC2(matchpoints1, matchpoints2, 9, (int) matchpoints1.size());

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    if (!transParameter.size()) {
        printf(" Perspective transformation matrix solution error! ");
        return;
    }
    printf(" Transformation matrix parameter: \n ");
    for (int i = 0; i < transParameter.size(); i++) {
        printf("%lf ", transParameter[i]);
        if ((i + 1) % 3 == 0) {
            printf("\n");
        }
    }
    printf("1\n");
    printf("Elapsed Time: %f\n", elapsed_time_ms);

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

    gls::Matrix<3, 3> homography = {
        { transParameter[0], transParameter[1], transParameter[2] },
        { transParameter[3], transParameter[4], transParameter[5] },
        { transParameter[6], transParameter[7], 1 }
    };

    gls::clRegisterAndFuse(glsContext, inputImage1, inputImage2, &outputImage, homography);

    auto outputImageCpu = outputImage.mapImage(CL_MAP_READ);
    outputImageCpu.write_png_file(PATH "fused.png", /*skip_alpha=*/ true);
    outputImage.unmapImage(outputImageCpu);
}

int main(int argc, const char * argv[]) {
    std::cout << "ImageRegistration Tests!\n";

    try {
        testSURF();
    } catch (const cl::Error& e) {
        std::cout << "cl::Error " << e.what() << " - " << gls::clStatusToString(e.err()) << std::endl;
    }

    return 0;
}
