//
//  main.cpp
//  ImageRegistration
//
//  Created by Fabio Riccardi on 8/26/22.
//

#include <iostream>

#include <filesystem>
#include <string>
#include <cmath>
#include <chrono>

#include "gls_logging.h"
#include "gls_cl_image.hpp"

#include "gls_linalg.hpp"

#include "SURF.hpp"

void testSURF() {
//    const auto srcImg1_ = gls::image<gls::rgb_pixel>::read_png_file("/Users/fabio/work/Image-Registration_SURF/3untagged.png");
//    const auto srcImg2_ = gls::image<gls::rgb_pixel>::read_png_file("/Users/fabio/work/Image-Registration_SURF/4untagged.png");

    const auto srcImg1_full = gls::image<gls::rgb_pixel>::read_jpeg_file("/Users/fabio/work/ImageRegistration/DSCF8601.jpg");
    const auto srcImg2_full = gls::image<gls::rgb_pixel>::read_jpeg_file("/Users/fabio/work/ImageRegistration/DSCF8603.jpg");

    const auto srcImg1_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg1_full, gls::rectangle {0, srcImg1_full->height/2, srcImg1_full->width, srcImg1_full->height/2});
    const auto srcImg2_ = std::make_unique<gls::image<gls::rgb_pixel>>(*srcImg2_full, gls::rectangle {0, srcImg2_full->height/2, srcImg2_full->width, srcImg2_full->height/2});

    gls::image<float> srcImg1(srcImg1_->width, srcImg1_->height);
    gls::image<float> srcImg2(srcImg2_->width, srcImg2_->height);

    srcImg1.apply([&](float *p, int x, int y) {
        const gls::rgb_pixel& pIn = (*srcImg1_)[y][x];
        *p = std::clamp(pIn.red * 0.299 + pIn.green * 0.587 + pIn.blue * 0.114, 0.0, 255.0);
    });

    srcImg2.apply([&](float *p, int x, int y) {
        const gls::rgb_pixel& pIn = (*srcImg2_)[y][x];
        *p = std::clamp(pIn.red * 0.299 + pIn.green * 0.587 + pIn.blue * 0.114, 0.0, 255.0);
    });

    std::vector<surf::Point2f> matchpoints1;
    std::vector<surf::Point2f> matchpoints2;

    auto t_start = std::chrono::high_resolution_clock::now();

    int matches_num = 150;
    bool isFeatureDection = surf::SURF_Detection(srcImg1, srcImg2, &matchpoints1, &matchpoints2, matches_num);
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

    testSURF();

    return 0;
}
