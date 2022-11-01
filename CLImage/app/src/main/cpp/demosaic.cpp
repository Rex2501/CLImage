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

#include "demosaic.hpp"

#include <iomanip>

#include "demosaic_cl.hpp"

#include "raw_converter.hpp"

#include "guided_filter.hpp"

#include "gls_logging.h"

const char* BayerPatternName[4] = {
    "GRBG",
    "GBRG",
    "RGGB",
    "BGGR"
};

static const char* TAG = "CLImage Pipeline";

gls::image<gls::rgb_pixel>::unique_ptr demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                     DemosaicParameters* demosaicParameters, bool calibrateFromImage) {
    gls::OpenCLContext glsContext("");

    auto rawConverter = std::make_unique<RawConverter>(&glsContext);

    auto t_start = std::chrono::high_resolution_clock::now();

    auto clsRGBImage = rawConverter->demosaicImage(rawImage, demosaicParameters, calibrateFromImage);

    auto rgbImage = RawConverter::convertToRGBImage(*clsRGBImage);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;

    return rgbImage;
}

gls::image<gls::rgb_pixel>::unique_ptr fastDemosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                          const DemosaicParameters& demosaicParameters) {
    gls::OpenCLContext glsContext("");

    auto rawConverter = std::make_unique<RawConverter>(&glsContext);

    auto t_start = std::chrono::high_resolution_clock::now();

    auto clsRGBImage = rawConverter->fastDemosaicImage(rawImage, demosaicParameters);

    auto rgbImage = RawConverter::convertToRGBImage(*clsRGBImage);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;

    return rgbImage;
}

template <int N>
bool inRange(const gls::DVector<N>& v, double minValue, double maxValue) {
    for (auto& e : v) {
        if (e < minValue || e > maxValue) {
            return false;
        }
    }
    return true;
}

YCbCrNLF BuildYCbCrNLF(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel_float>& image) {
    gls::cl_image_2d<gls::rgba_pixel_float> noiseStats(glsContext->clContext(), image.width, image.height);
    applyKernel(glsContext, "noiseStatistics", image, &noiseStats);
    const auto noiseStatsCpu = noiseStats.mapImage();

    // Only consider pixels with variance lower than the expected noise value
    double varianceMax = 0.001;

    // Limit to pixels the more linear intensity zone of the sensor
    const double maxValue = 0.5;
    const double minValue = 0.001;

    // Collect pixel statistics
    double s_x = 0;
    double s_xx = 0;
    gls::DVector<3> s_y = gls::DVector<3>::zeros();
    gls::DVector<3> s_xy = gls::DVector<3>::zeros();

    double N = 0;
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = { ns[1], ns[2], ns[3] };

        if (m >= minValue && m <= maxValue && inRange<3>(v, 0, varianceMax)) {
            s_x += m;
            s_y += v;
            s_xx += m * m;
            s_xy += m * v;
            N++;
        }
    });

    // Linear regression on pixel statistics to extract a linear noise model: nlf = A + B * Y
    auto nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    auto nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    // Estimate regression mean square error
    gls::DVector<3> err2 = gls::DVector<3>::zeros();
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = { ns[1], ns[2], ns[3] };

        if (m >= minValue && m <= maxValue && inRange<3>(v, 0, varianceMax)) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            err2 += diff * diff;
        }
    });
    err2 /= N;

//    std::cout << "1) Pyramid NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(err2)
//              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels"<< std::endl;

    // Redo the statistics collection limiting the sample to pixels that fit well the linear model
    s_x = 0;
    s_xx = 0;
    s_y = gls::DVector<3>::zeros();
    s_xy = gls::DVector<3>::zeros();
    N = 0;
    gls::DVector<3> newErr2 = gls::DVector<3>::zeros();
    int discarded = 0;
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = { ns[1], ns[2], ns[3] };

        if (m >= minValue && m <= maxValue && inRange<3>(v, 0, varianceMax)) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = abs(nlfP - v);
            auto diffSquare = diff * diff;

            if (all(diffSquare <= 0.5 * err2)) {
                s_x += m;
                s_y += v;
                s_xx += m * m;
                s_xy += m * v;
                N++;
                newErr2 += diffSquare;
            } else {
                discarded++;
            }
        }
    });
    newErr2 /= N;

    // Estimate the new regression parameters
    nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    assert(all(newErr2 < err2));

    std::cout << "Pyramid NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(newErr2)
              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels" << std::endl;

    noiseStats.unmapImage(noiseStatsCpu);

    return std::pair (
        gls::Vector<3> { (float) nlfA[0], (float) nlfA[1], (float) nlfA[2] }, // A values
        gls::Vector<3> { (float) nlfB[0], (float) nlfB[1], (float) nlfB[2] }  // B values
    );
}

RawNLF BuildRawNLF(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::luma_pixel_float>& rawImage, BayerPattern bayerPattern) {
    gls::cl_image_2d<gls::rgba_pixel_float> meanImage(glsContext->clContext(), rawImage.width / 2, rawImage.height / 2);
    gls::cl_image_2d<gls::rgba_pixel_float> varImage(glsContext->clContext(), rawImage.width / 2, rawImage.height / 2);

    rawNoiseStatistics(glsContext, rawImage, bayerPattern, &meanImage, &varImage);

    const auto meanImageCpu = meanImage.mapImage();
    const auto varImageCpu = varImage.mapImage();

    // Only consider pixels with variance lower than the expected noise value
    double varianceMax = 0.001;

    // Limit to pixels the more linear intensity zone of the sensor
    const double maxValue = 0.5;
    const double minValue = 0.001;

    // Collect pixel statistics
    gls::DVector<4> s_x = gls::DVector<4>::zeros();
    gls::DVector<4> s_y = gls::DVector<4>::zeros();
    gls::DVector<4> s_xx = gls::DVector<4>::zeros();
    gls::DVector<4> s_xy = gls::DVector<4>::zeros();

    double N = 0;
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = { mm[0], mm[1], mm[2], mm[3] };
        gls::DVector<4> v = { vv[0], vv[1], vv[2], vv[3] };

        if (inRange<4>(m, minValue, maxValue) && inRange<4>(v, 0, varianceMax)) {
            s_x += m;
            s_y += v;
            s_xx += m * m;
            s_xy += m * v;
            N++;
        }
    });

    // Linear regression on pixel statistics to extract a linear noise model: nlf = A + B * Y
    auto nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    auto nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    // Estimate regression mean square error
    gls::DVector<4> err2 = gls::DVector<4>::zeros();
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = { mm[0], mm[1], mm[2], mm[3] };
        gls::DVector<4> v = { vv[0], vv[1], vv[2], vv[3] };

        if (inRange<4>(m, minValue, maxValue) && inRange<4>(v, 0, varianceMax)) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            err2 += diff * diff;
        }
    });
    err2 /= N;

//    std::cout << "RAW NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(err2)
//              << " on " << std::setprecision(1) << std::fixed << 100 * N / (rawImage.width * rawImage.height) << "% pixels"<< std::endl;

    // Redo the statistics collection limiting the sample to pixels that fit well the linear model
    s_x = gls::DVector<4>::zeros();
    s_y = gls::DVector<4>::zeros();
    s_xx = gls::DVector<4>::zeros();
    s_xy = gls::DVector<4>::zeros();
    N = 0;
    gls::DVector<4> newErr2 = gls::DVector<4>::zeros();
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = { mm[0], mm[1], mm[2], mm[3] };
        gls::DVector<4> v = { vv[0], vv[1], vv[2], vv[3] };

        if (inRange<4>(m, minValue, maxValue) && inRange<4>(v, 0, varianceMax)) {
            const auto nlfP = nlfA + nlfB * m;
            const auto diff = abs(nlfP - v);
            const auto diffSquare = diff * diff;

            if (all(diffSquare <= 0.5 * err2)) {
                s_x += m;
                s_y += v;
                s_xx += m * m;
                s_xy += m * v;
                N++;
                newErr2 += diffSquare;
            }
        }
    });
    newErr2 /= N;

    // Estimate the new regression parameters
    nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    assert(all(newErr2 < err2));

    std::cout << "RAW NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", MSE: " << sqrt(newErr2)
              << " on " << std::setprecision(1) << std::fixed << 100 * N / (rawImage.width * rawImage.height) << "% pixels"<< std::endl;

    meanImage.unmapImage(meanImageCpu);
    varImage.unmapImage(varImageCpu);

    return std::pair (
        gls::Vector<4> { (float) nlfA[0], (float) nlfA[1], (float) nlfA[2], (float) nlfA[3] }, // A values
        gls::Vector<4> { (float) nlfB[0], (float) nlfB[1], (float) nlfB[2], (float) nlfB[3] }  // B values
    );
}
