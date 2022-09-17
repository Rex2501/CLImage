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

void testSURF() {
    auto glsContext = new gls::OpenCLContext("");

//    const auto srcImg1_ = gls::image<gls::rgb_pixel>::read_png_file("/Users/fabio/work/Image-Registration_SURF/3untagged.png");
//    const auto srcImg2_ = gls::image<gls::rgb_pixel>::read_png_file("/Users/fabio/work/Image-Registration_SURF/4untagged.png");

#if __APPLE__
    const auto srcImg1_full = gls::image<gls::rgb_pixel>::read_jpeg_file("/Users/fabio/work/ImageRegistration/DSCF8601.jpg");
    const auto srcImg2_full = gls::image<gls::rgb_pixel>::read_jpeg_file("/Users/fabio/work/ImageRegistration/DSCF8603.jpg");
#else
    const auto srcImg1_full = gls::image<gls::rgb_pixel>::read_jpeg_file("/data/local/tmp/DSCF8601.jpg");
    const auto srcImg2_full = gls::image<gls::rgb_pixel>::read_jpeg_file("/data/local/tmp/DSCF8603.jpg");
#endif

    const auto srcImg1_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg1_full, gls::rectangle {0, srcImg1_full->height/2, srcImg1_full->width, srcImg1_full->height/2});
    const auto srcImg2_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg2_full, gls::rectangle {0, srcImg2_full->height/2, srcImg2_full->width, srcImg2_full->height/2});

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

    const auto test = gls::cl_image_buffer_2d<float>(glsContext->clContext(), srcImg1_->width, srcImg1_->height);
    test.copyPixelsFrom(srcImg1);

//    {
//        gls::cl_image_2d<gls::luma_alpha_pixel_fp32> input(glsContext->clContext(), srcImg1.width, srcImg1.height);
//        gls::cl_image_2d<gls::luma_pixel_fp32> output(glsContext->clContext(), srcImg1.width, srcImg1.height);
//
//        auto inputCpu = input.mapImage();
//        inputCpu.apply([&srcImg1](gls::luma_alpha_pixel_fp32 *p, int x, int y) {
//            *p = { srcImg1[y][x], 1 };
//        });
//        input.unmapImage(inputCpu);
//
//        boxBlurScan(glsContext, input, &output, 10);
//
//        const auto blurredImage = output.mapImage();
//        gls::image<gls::luma_pixel> blurredImageLuma(blurredImage.width, blurredImage.height);
//
//        blurredImageLuma.apply([&blurredImage](gls::luma_pixel* p, int x, int y){
//            *p = std::clamp(blurredImage[y][x].luma, 0.0f, 255.0f);
//        });
//
//        blurredImageLuma.write_png_file("/Users/fabio/blurred.png");
//        output.unmapImage(blurredImage);
//    }

    std::vector<surf::Point2f> matchpoints1;
    std::vector<surf::Point2f> matchpoints2;

    auto t_start = std::chrono::high_resolution_clock::now();

    int matches_num = 150;
    bool isFeatureDection = surf::SURF_Detection(glsContext, srcImg1, srcImg2, &matchpoints1, &matchpoints2, matches_num);
    printf("isFeatureDection: %d\n", isFeatureDection);

    std::vector<float> transParameter;
    transParameter = surf::getRANSAC2(matchpoints1, matchpoints2, 9, 2 * 150);

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
