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

#include "pyramidal_denoise.hpp"

#include <iomanip>

template
PyramidalDenoise<5>::PyramidalDenoise(gls::OpenCLContext* glsContext, int width, int height, DenoiseAlgorithm denoiseAlgorithm);

template
typename PyramidalDenoise<5>::imageType* PyramidalDenoise<5>::denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, 5>* denoiseParameters,
                                                                      imageType* image, const gls::Matrix<3, 3>& rgb_cam,
                                                                      const gls::rectangle* gmb_position, bool rotate_180,
                                                                      gls::Matrix<5, 6>* nlfParameters);

struct BilateralDenoiser : ImageDenoiser {
    BilateralDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                 const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                 const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                 float chromaBoost, int pyramidLevel,
              gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImage(glsContext, inputImage, var_a, var_b, chromaBoost, pyramidLevel == 0 ? true : false, outputImage);
    }
};

struct GuidedFastDenoiser : ImageDenoiser {
    GuidedFastDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                 const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                 const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                 float chromaBoost, int pyramidLevel,
                 gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImageGuided(glsContext, inputImage, var_a, var_b, outputImage);
    }
};

struct GuidedPreciseDenoiser : ImageDenoiser {
    GuidedFilter guidedFilter;

    GuidedPreciseDenoiser(gls::OpenCLContext* glsContext, int width, int height) :
        ImageDenoiser(glsContext, width, height),
        guidedFilter(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                 const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                 const gls::Vector<3>& var_a, const gls::Vector<3>& var_b,
                 float chromaBoost, int pyramidLevel,
                 gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        guidedFilter.filter(glsContext, inputImage, /*filterSize=*/ 5, var_b, outputImage);
    }
};

template <size_t levels>
PyramidalDenoise<levels>::PyramidalDenoise(gls::OpenCLContext* glsContext, int width, int height, DenoiseAlgorithm denoiseAlgorithm) {
    for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
        imagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
    }
    for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
        denoisedImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);

        switch (denoiseAlgorithm) {
            case Bilateral:
                denoiser[i] = std::make_unique<BilateralDenoiser>(glsContext, width/scale, height/scale);
                break;
            case GuidedFast:
                if (i < levels - 1) {
                    denoiser[i] = std::make_unique<GuidedFastDenoiser>(glsContext, width/scale, height/scale);
                } else {
                    denoiser[i] = std::make_unique<BilateralDenoiser>(glsContext, width/scale, height/scale);
                }
                break;
            case GuidedPrecise:
                denoiser[i] = std::make_unique<GuidedPreciseDenoiser>(glsContext, width/scale, height/scale);
                break;
        }
    }
}

gls::Vector<6> computeNoiseStatistics(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel_float>& image);

gls::Vector<6> nflMultiplier(const DenoiseParameters &denoiseParameters) {
    float luma_mul = denoiseParameters.luma * denoiseParameters.luma;
    float chroma_mul = denoiseParameters.chroma * denoiseParameters.chroma;
    return { luma_mul, chroma_mul, chroma_mul, luma_mul, chroma_mul, chroma_mul };
}

template <size_t levels>
typename PyramidalDenoise<levels>::imageType* PyramidalDenoise<levels>::denoise(gls::OpenCLContext* glsContext,
                                                                                std::array<DenoiseParameters, levels>* denoiseParameters,
                                                                                imageType* image, const gls::Matrix<3, 3>& rgb_cam,
                                                                                const gls::rectangle* gmb_position, bool rotate_180,
                                                                                gls::Matrix<levels, 6>* nlfParameters) {
    const bool calibrate_nlf = false; // gmb_position != nullptr;

    std::array<std::array<float, 6>, levels> calibrated_nlf;

    for (int i = 0; i < levels; i++) {
        if (i < levels - 1) {
            resampleImage(glsContext, "downsampleImage", i == 0 ? *image : *imagePyramid[i - 1], imagePyramid[i].get());
        }

        if (calibrate_nlf) {
            (*nlfParameters)[i] = computeNoiseStatistics(glsContext, i == 0 ? *image : *(imagePyramid[i - 1]));
        }

        calibrated_nlf[i] = (*nlfParameters)[i] * nflMultiplier((*denoiseParameters)[i]);
    }

    // Denoise the bottom of the image pyramid
    const auto& np = calibrated_nlf[levels-1];
    denoiser[levels-1]->denoise(glsContext, *(imagePyramid[levels-2]),
                                { np[0], np[1], np[2] }, { np[3], np[4], np[5] },
                                (*denoiseParameters)[levels-1].chromaBoost, /*pyramidLevel=*/ levels-1,
                                denoisedImagePyramid[levels-1].get());

    for (int i = levels - 2; i >= 0; i--) {
        gls::cl_image_2d<gls::rgba_pixel_float>* denoiseInput = i > 0 ? imagePyramid[i - 1].get() : image;

        // Denoise current layer
        const auto& np = calibrated_nlf[i];
        denoiser[i]->denoise(glsContext, *denoiseInput,
                             { np[0], np[1], np[2] }, { np[3], np[4], np[5] },
                             (*denoiseParameters)[i].chromaBoost, /*pyramidLevel=*/ i,
                             denoisedImagePyramid[i].get());

        std::cout << "Reassembling layer " << i << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;
        // Subtract noise from previous layer
        reassembleImage(glsContext, *(denoisedImagePyramid[i]), *(imagePyramid[i]), *(denoisedImagePyramid[i+1]),
                        (*denoiseParameters)[i].sharpening, {np[0], np[3]}, denoisedImagePyramid[i].get());
    }

    return denoisedImagePyramid[0].get();
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

gls::Vector<6> computeNoiseStatistics(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel_float>& image) {
    gls::cl_image_2d<gls::rgba_pixel_float> noiseStats(glsContext->clContext(), image.width, image.height);
    applyKernel(glsContext, "noiseStatistics", image, &noiseStats);
    const auto noiseStatsCpu = noiseStats.mapImage();

    const double maxVarianceHistVal = 0.01;
    int minBinSize = 0.0015 * image.width  * image.height;
    const int variance_bins = 128;
    uint32_t log_variance[variance_bins] = { 0 };
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns) {
        int hist_index = rint((variance_bins - 1) * fmin(log(1 + ns[1]) / log(1 + maxVarianceHistVal), 1));
        log_variance[hist_index]++;
    });

    // Only consider pixels with variance lower than the expected noise value
    double varianceMax = 0.001;

    std::cout << "log_variance: ";
    int max_bin = 0;
    int max_val = 0;
    bool descending = false;
    for (int i = 0; i < variance_bins; i++) {
        if (log_variance[i] > max_val) {
            max_bin = i;
            max_val = log_variance[i];
            descending = false;
        } else if (log_variance[i] > minBinSize && log_variance[i] <= max_val) {
            varianceMax = 40 * (exp((i / ((float) variance_bins - 1)) * log(1 + maxVarianceHistVal)) - 1);
            std::cout << i << " - ";
            std::cout << log_variance[i] << " @ " << std::scientific << varianceMax << ", ";
            break;
        }
    }
    std::cout << std::endl;

    // Limit to pixels the more linear intensity zone of the sensor
    const double maxValue = 0.5;
    const double minValue = 0.001;

    // Collect pixel statistics
    gls::DVector<3> s_x = {{ 0, 0, 0 }};
    gls::DVector<3> s_y = {{ 0, 0, 0 }};
    gls::DVector<3> s_xx = {{ 0, 0, 0 }};
    gls::DVector<3> s_xy = {{ 0, 0, 0 }};

    double N = 0;
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = {{ ns[1], ns[2], ns[3] }};

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
    gls::DVector<3> err2 = {{ 0, 0, 0 }};
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = {{ ns[1], ns[2], ns[3] }};

        if (m >= minValue && m <= maxValue && inRange<3>(v, 0, varianceMax)) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            err2 += diff * diff;
        }
    });
    err2 = sqrt(err2 / N);

//    std::cout << "Pyramid NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", err2: " << err2
//              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels"<< std::endl;

    // Redo the statistics collection limiting the sample to pixels that fit well the linear model
    s_x = {{ 0, 0, 0 }};
    s_y = {{ 0, 0, 0 }};
    s_xx = {{ 0, 0, 0 }};
    s_xy = {{ 0, 0, 0 }};
    N = 0;
    gls::DVector<3> newErr2 = {{ 0, 0, 0 }};
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& ns, int x, int y) {
        double m = ns[0];
        gls::DVector<3> v = {{ ns[1], ns[2], ns[3] }};

        if (m >= minValue && m <= maxValue && inRange<3>(v, 0, varianceMax)) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            diff *= diff;

            if (diff[0] <= err2[0] && diff[1] <= err2[1] && diff[2] <= err2[2]) {
                s_x += m;
                s_y += v;
                s_xx += m * m;
                s_xy += m * v;
                N++;
                newErr2 += diff;
            }
        }
    });
    newErr2 = sqrt(newErr2 / N);

    // Estimate the new regression parameters
    nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    assert(err2[0] >= newErr2[0] && err2[1] >= newErr2[1] && err2[2] >= newErr2[2]);

    std::cout << "Pyramid NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", err2: " << newErr2
              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels"<< std::endl;

    noiseStats.unmapImage(noiseStatsCpu);

    return {
        (float) nlfA[0], (float) nlfA[1], (float) nlfA[2], // A values
        (float) nlfB[0], (float) nlfB[1], (float) nlfB[2]  // B values
    };
}

gls::Vector<8> computeRawNoiseStatistics(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::luma_pixel_float>& rawImage, BayerPattern bayerPattern) {
    gls::cl_image_2d<gls::rgba_pixel_float> meanImage(glsContext->clContext(), rawImage.width / 2, rawImage.height / 2);
    gls::cl_image_2d<gls::rgba_pixel_float> varImage(glsContext->clContext(), rawImage.width / 2, rawImage.height / 2);

    rawNoiseStatistics(glsContext, rawImage, bayerPattern, &meanImage, &varImage);

    const auto meanImageCpu = meanImage.mapImage();
    const auto varImageCpu = varImage.mapImage();

    // Limit to pixels the more linear intensity zone of the sensor
    const double maxValue = 0.5;
    const double minValue = 0.001;

    // Collect pixel statistics
    gls::DVector<4> s_x = {{ 0, 0, 0, 0 }};
    gls::DVector<4> s_y = {{ 0, 0, 0, 0 }};
    gls::DVector<4> s_xx = {{ 0, 0, 0, 0 }};
    gls::DVector<4> s_xy = {{ 0, 0, 0, 0 }};

    const int variance_bins = 128;
    uint32_t log_variance[variance_bins] = { 0 };
    const double maxVarianceHistVal = 0.01;

    // auto greenVarImageCpu = gls::image<gls::luma_pixel_16>(varImageCpu.width, varImageCpu.height);

    varImageCpu.apply([&](const gls::rgba_pixel_float& vv, int x, int y) {
        int hist_index = rint((variance_bins - 1) * fmin(log(1 + vv[1]) / log(1 + maxVarianceHistVal), 1));
        log_variance[hist_index]++;
        // greenVarImageCpu[y][x] = 0xffff * vv[1] / meanImageCpu[y][x][1];
    });

    // greenVarImageCpu.write_png_file("/Users/fabio/greenVar.png");

    // Only consider pixels with variance lower than the expected noise value
    double varianceMax = 0.001;

    std::cout << "log_variance: ";
    int max_bin = 0;
    int max_val = 0;
    int minBinSize = 0.0004 * rawImage.width * rawImage.height;
    bool descending = false;
    for (int i = 0; i < variance_bins; i++) {
        if (log_variance[i] > max_val) {
            max_bin = i;
            max_val = log_variance[i];
            descending = false;
        } else if (log_variance[i] > minBinSize && log_variance[i] <= max_val) {
            varianceMax = 40 * (exp((i / ((float) variance_bins - 1)) * log(1 + maxVarianceHistVal)) - 1);
            std::cout << i << " - ";
            std::cout << log_variance[i] << " @ " << std::scientific << varianceMax << ", ";
            break;
        }
    }
    std::cout << std::endl;

    double N = 0;
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = {{ mm[0], mm[1], mm[2], mm[3] }};
        gls::DVector<4> v = {{ vv[0], vv[1], vv[2], vv[3] }};

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
    gls::DVector<4> err2 = {{ 0, 0, 0, 0 }};
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = {{ mm[0], mm[1], mm[2], mm[3] }};
        gls::DVector<4> v = {{ vv[0], vv[1], vv[2], vv[3] }};

        if (inRange<4>(m, minValue, maxValue) && inRange<4>(v, 0, varianceMax)) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            err2 += diff * diff;
        }
    });
    err2 = sqrt(err2 / N);

//    std::cout << "RAW NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", err2: " << err2
//              << " on " << std::setprecision(1) << std::fixed << 100 * N / (rawImage.width * rawImage.height) << "% pixels"<< std::endl;

    // Redo the statistics collection limiting the sample to pixels that fit well the linear model
    s_x = {{ 0, 0, 0, 0 }};
    s_y = {{ 0, 0, 0, 0 }};
    s_xx = {{ 0, 0, 0, 0 }};
    s_xy = {{ 0, 0, 0, 0 }};
    N = 0;
    gls::DVector<4> newErr2 = {{ 0, 0, 0, 0 }};
    meanImageCpu.apply([&](const gls::rgba_pixel_float& mm, int x, int y) {
        const gls::rgba_pixel_float& vv = varImageCpu[y][x];
        gls::DVector<4> m = {{ mm[0], mm[1], mm[2], mm[3] }};
        gls::DVector<4> v = {{ vv[0], vv[1], vv[2], vv[3] }};

        if (inRange<4>(m, minValue, maxValue) && inRange<4>(v, 0, varianceMax)) {
            auto nlfP = nlfA + nlfB * m;
            auto diff = nlfP - v;
            diff *= diff;

            if (diff[0] <= err2[0] && diff[1] <= err2[1] && diff[2] <= err2[2] && diff[3] <= err2[3]) {
                s_x += m;
                s_y += v;
                s_xx += m * m;
                s_xy += m * v;
                N++;
                newErr2 += diff;
            }
        }
    });
    newErr2 = sqrt(newErr2 / N);

    // Estimate the new regression parameters
    nlfB = max((N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x), 1e-8);
    nlfA = max((s_y - nlfB * s_x) / N, 1e-8);

    // assert(err2[0] >= newErr2[0] && err2[1] >= newErr2[1] && err2[2] >= newErr2[2] && err2[3] >= newErr2[3]);

    std::cout << "RAW NLF A: " << std::setprecision(4) << std::scientific << nlfA << ", B: " << nlfB << ", err2: " << newErr2
              << " on " << std::setprecision(1) << std::fixed << 100 * N / (rawImage.width * rawImage.height) << "% pixels"<< std::endl;

    meanImage.unmapImage(meanImageCpu);
    varImage.unmapImage(varImageCpu);

    return {
        (float) nlfA[0], (float) nlfA[1], (float) nlfA[2], (float) nlfA[3], // A values
        (float) nlfB[0], (float) nlfB[1], (float) nlfB[2], (float) nlfB[3]  // B values
    };
}
