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

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#else
#define half float
#define half2 float2
#define half3 float3
#define half4 float4
#define read_imageh read_imagef
#define write_imageh write_imagef
#define HALF_MAX MAXFLOAT
#define convert_half2(val)    (val)
#define convert_half3(val)    (val)
#define convert_half4(val)    (val)
#endif

//#define LENS_SHADING true
//#define LENS_SHADING_GAIN 1

enum BayerPattern {
    grbg = 0,
    gbrg = 1,
    rggb = 2,
    bggr = 3
};

enum { raw_red = 0, raw_green = 1, raw_blue = 2, raw_green2 = 3 };

constant const int2 bayerOffsets[4][4] = {
    { {1, 0}, {0, 0}, {0, 1}, {1, 1} }, // grbg
    { {0, 1}, {0, 0}, {1, 0}, {1, 1} }, // gbrg
    { {0, 0}, {0, 1}, {1, 1}, {1, 0} }, // rggb
    { {1, 1}, {0, 1}, {0, 0}, {1, 0} }  // bggr
};

#if defined(__QCOMM_QGPU_A3X__) || \
    defined(__QCOMM_QGPU_A4X__) || \
    defined(__QCOMM_QGPU_A5X__) || \
    defined(__QCOMM_QGPU_A6X__) || \
    defined(__QCOMM_QGPU_A7V__) || \
    defined(__QCOMM_QGPU_A7P__)

// Qualcomm's smoothstep implementation can be really slow...

#define smoothstep(edge0, edge1, x) \
   ({ typedef __typeof__ (x) type_of_x; \
      type_of_x _edge0 = (edge0); \
      type_of_x _edge1 = (edge1); \
      type_of_x _x = (x); \
      type_of_x t = clamp((_x - _edge0) / (_edge1 - _edge0), (type_of_x) 0, (type_of_x) 1); \
      t * t * (3 - 2 * t); })

#endif

// Apple's half float fail to compile with the system's min/max functions

#undef min
#undef max
#undef abs

#define min(a, b) ({__typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;})

#define max(a, b) ({__typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;})

#define abs(a) ({__typeof__(a) _a = (a); \
    _a > 0 ? _a : -_a;})

#ifdef __APPLE__
inline half2 __attribute__((overloadable)) myconvert_half2(float2 val) {
    return (half2) (val.x, val.y);
}

inline half3 __attribute__((overloadable)) myconvert_half3(float3 val) {
    return (half3) (val.x, val.y, val.z);
}

inline half4 __attribute__((overloadable)) myconvert_half4(float4 val) {
    return (half4) (val.x, val.y, val.z, val.w);
}

#define convert_half2(val)    myconvert_half2(val)
#define convert_half3(val)    myconvert_half3(val)
#define convert_half4(val)    myconvert_half4(val)
#endif

// Fast 5x5 box filtering with linear subsampling
typedef struct ConvolutionParameters {
    float weight;
    float2 offset;
} ConvolutionParameters;

constant ConvolutionParameters boxFilter5x5[9] = {
    { 0.1600, { -1.5000, -1.5000 } },
    { 0.1600, {  0.5000, -1.5000 } },
    { 0.0800, {  2.0000, -1.5000 } },
    { 0.1600, { -1.5000,  0.5000 } },
    { 0.1600, {  0.5000,  0.5000 } },
    { 0.0800, {  2.0000,  0.5000 } },
    { 0.0800, { -1.5000,  2.0000 } },
    { 0.0800, {  0.5000,  2.0000 } },
    { 0.0400, {  2.0000,  2.0000 } },
};

constant half gaussianBlur5x5[5][5] = {
    { 1.8316e-02, 8.2085e-02, 1.3534e-01, 8.2085e-02, 1.8316e-02 },
    { 8.2085e-02, 3.6788e-01, 6.0653e-01, 3.6788e-01, 8.2085e-02 },
    { 1.3534e-01, 6.0653e-01, 1.0000e+00, 6.0653e-01, 1.3534e-01 },
    { 8.2085e-02, 3.6788e-01, 6.0653e-01, 3.6788e-01, 8.2085e-02 },
    { 1.8316e-02, 8.2085e-02, 1.3534e-01, 8.2085e-02, 1.8316e-02 }
};

constant half gaussianBlur3x3[3][3] = {
    { 1/16.0, 1/8.0, 1/16.0 },
    { 1/8.0,  1/4.0, 1/8.0  },
    { 1/16.0, 1/8.0, 1/16.0 }
};

constant float2 sobelKernel2D[3][3] = {
    { { 1,  1 }, { 0,  2 }, { -1,  1 } },
    { { 2,  0 }, { 0,  0 }, { -2,  0 } },
    { { 1, -1 }, { 0, -2 }, { -1, -1 } },
};


// Work on one Quad (2x2) at a time
kernel void scaleRawData(read_only image2d_t rawImage, write_only image2d_t scaledRawImage,
                         int bayerPattern, float4 vScaleMul, float blackLevel) {
    float *scaleMul = (float *) &vScaleMul;
    const int2 imageCoordinates = (int2) (2 * get_global_id(0), 2 * get_global_id(1));
    for (int c = 0; c < 4; c++) {
        int2 o = bayerOffsets[bayerPattern][c];
        write_imagef(scaledRawImage, imageCoordinates + (int2) (o.x, o.y),
                     max(scaleMul[c] * (read_imagef(rawImage, imageCoordinates + (int2) (o.x, o.y)).x - blackLevel), 0.0f));
    }
}

float2 sobel(read_only image2d_t inputImage, int x, int y) {
    float2 value = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            float sample = read_imagef(inputImage, (int2)(x + i, y + j)).x;
            value += sobelKernel2D[j+1][i+1] * sample;
        }
    }

    return value / sqrt(4.5);
}

float2 gaussFilteredSobel3x3(read_only image2d_t inputImage, int x, int y) {
    // Average Sobel Filter on a 3x3 raw patch
    float2 sum = 0;
    float2 absSum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            float2 sample = gaussianBlur3x3[j + 1][i + 1] * sobel(inputImage, x + i, y + j);
            sum += sample;
            absSum += abs(sample);
        }
    }
    // return sum;
    return copysign(absSum, sum);
}

float2 gaussFilteredSobel5x5(read_only image2d_t inputImage, int x, int y) {
    // Average Sobel Filter on a 5x5 raw patch
    float2 sum = 0;
    float2 absSum = 0;
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            float2 sample = gaussianBlur5x5[j + 2][i + 2] * sobel(inputImage, x + i, y + j);
            sum += sample;
            absSum += abs(sample);
        }
    }
    sum /= 4;
    absSum /= 4;
    // return sum;
    return copysign(absSum, sum);
}

kernel void rawImageGradient(read_only image2d_t inputImage, write_only image2d_t gradientImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    float2 gradient = gaussFilteredSobel3x3(inputImage, imageCoordinates.x, imageCoordinates.y);

    write_imagef(gradientImage, imageCoordinates, (float4) (gradient, 0, 0));
}

kernel void rawImageSobel(read_only image2d_t inputImage, write_only image2d_t gradientImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    float2 gradient = sobel(inputImage, imageCoordinates.x, imageCoordinates.y);

    write_imagef(gradientImage, imageCoordinates, (float4) (gradient, abs(gradient)));
}

// Modified Hamilton-Adams green channel interpolation

constant const float kHighNoiseVariance = 1e-3;

#define RAW(i, j) read_imagef(rawImage, (int2)(imageCoordinates.x + i, imageCoordinates.y + j)).x

kernel void interpolateGreen(read_only image2d_t rawImage,
                             read_only image2d_t gradientImage,
                             write_only image2d_t greenImage,
                             int bayerPattern, float2 greenVariance) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const int x = imageCoordinates.x;
    const int y = imageCoordinates.y;

    const int2 r = bayerOffsets[bayerPattern][raw_red];
    const int2 b = bayerOffsets[bayerPattern][raw_blue];

    const bool red_pixel = (r.x & 1) == (x & 1) && (r.y & 1) == (y & 1);
    const bool blue_pixel = (b.x & 1) == (x & 1) && (b.y & 1) == (y & 1);

    const float lowNoise = 1 - smoothstep(3.5e-4, 2e-3, greenVariance.y);

    if (red_pixel || blue_pixel) {
        // Red and Blue pixel locations
        float g_left  = RAW(-1, 0);
        float g_right = RAW(1, 0);
        float g_up    = RAW(0, -1);
        float g_down  = RAW(0, 1);

        float c_xy    = RAW(0, 0);

        float c_left  = RAW(-2, 0);
        float c_right = RAW(2, 0);
        float c_up    = RAW(0, -2);
        float c_down  = RAW(0, 2);

        float c2_top_left = RAW(-1, -1);
        float c2_top_right = RAW(1, -1);
        float c2_bottom_left = RAW(-1, 1);
        float c2_bottom_right = RAW(1, 1);
        float c2_ave = (c2_top_left + c2_top_right + c2_bottom_left + c2_bottom_right) / 4;

        // Estimate gradient intensity and direction
        float g_ave = (g_left + g_right + g_up + g_down) / 4;
        float2 gradient = abs(read_imagef(gradientImage, imageCoordinates).xy);

        // Hamilton-Adams second order Laplacian Interpolation
        float2 g_lf = { (g_left + g_right) / 2, (g_up + g_down) / 2 };
        float2 g_hf = { ((c_left + c_right) - 2 * c_xy) / 4, ((c_up + c_down) - 2 * c_xy) / 4 };

        // Minimum gradient threshold wrt the noise model
        float rawStdDev = sqrt(greenVariance.x + greenVariance.y * g_ave);
        float gradient_threshold = smoothstep(rawStdDev, 4 * rawStdDev, length(gradient));
        float low_gradient_threshold = 1 - smoothstep(2 * rawStdDev, 8 * rawStdDev, length(gradient));

        // Sharpen low contrast areas
        float sharpening = (0.5 + 0.5 * lowNoise) * gradient_threshold * (1 + lowNoise * low_gradient_threshold);

        // Edges that are in strong highlights tend to exagerate the gradient
        float highlights_edge = 1 - smoothstep(0.25, 1.0, max(c_xy, max(max(c_left, c_right), max(c_up, c_down))));

        // Gradient direction in [0..1]
        float direction = 2 * atan2pi(gradient.y, gradient.x);

        if (greenVariance.y < kHighNoiseVariance) {
            // Bias result towards vertical and horizontal lines
            direction = direction < 0.5 ? mix(direction, 0, 1 - smoothstep(0.3, 0.45, direction))
                                        : mix(direction, 1, smoothstep((1 - 0.45), (1 - 0.3), direction));
        }

        // TODO: Doesn't seem like a good idea, maybe for high noise images?
        // If the gradient is below threshold interpolate against the grain
        // direction = mix(1 - direction, direction, gradient_threshold);

        // Estimate the degree of correlation between channels to drive the amount of HF extraction
        const float cmin = min(c_xy, min(g_ave, c2_ave));
        const float cmax = max(c_xy, max(g_ave, c2_ave));
        float whiteness = cmin / cmax;

        // Modulate the HF component of the reconstructed green using the whiteness and the gradient magnitude
        float2 g_est = g_lf - highlights_edge * whiteness * sharpening * g_hf;

        // Green pixel estimation
        float green = mix(g_est.y, g_est.x, direction);

        // Limit the range of HF correction to something reasonable
        float max_overshoot = mix(1.0, 1.5, whiteness);
        float min_overshoot = mix(1.0, 0.5, whiteness);

        float gmax = max(max(g_left, g_right), max(g_up, g_down));
        float gmin = min(min(g_left, g_right), min(g_up, g_down));
        green = clamp(green, min_overshoot * gmin, max_overshoot * gmax);

        write_imagef(greenImage, imageCoordinates, clamp(green, 0.0, 1.0));
    } else {
        // Green pixel locations
        write_imagef(greenImage, imageCoordinates, read_imagef(rawImage, imageCoordinates).x);
    }
}

/*
    Modified Hamilton-Adams red-blue channels interpolation: Red and Blue locations are interpolate first,
    Green locations are interpolated next as they use the data from the previous step
*/

#define GREEN(i, j) read_imagef(greenImage, (int2)(imageCoordinates.x + i, imageCoordinates.y + j)).x

// Interpolate the other color at Red and Blue RAW locations

void interpolateRedBluePixel(read_only image2d_t rawImage,
                             read_only image2d_t greenImage,
                             read_only image2d_t gradientImage,
                             write_only image2d_t rgbImage,
                             float2 redVariance, float2 blueVariance,
                             bool red_pixel, int2 imageCoordinates) {
    float green = GREEN(0, 0);
    float c1 = RAW(0, 0);

    float g_top_left      = GREEN(-1, -1);
    float g_top_right     = GREEN(1, -1);
    float g_bottom_left   = GREEN(-1, 1);
    float g_bottom_right  = GREEN(1, 1);

    float c2_top_left     = RAW(-1, -1);
    float c2_top_right    = RAW(1, -1);
    float c2_bottom_left  = RAW(-1, 1);
    float c2_bottom_right = RAW(1, 1);
    float c2_ave = (c2_top_left + c2_top_right + c2_bottom_left + c2_bottom_right) / 4;

    float gc_top_left     = g_top_left     - c2_top_left;
    float gc_top_right    = g_top_right    - c2_top_right;
    float gc_bottom_left  = g_bottom_left  - c2_bottom_left;
    float gc_bottom_right = g_bottom_right - c2_bottom_right;

    float g_top_left2      = GREEN(-2, -2);
    float g_top_right2     = GREEN(2, -2);
    float g_bottom_left2   = GREEN(-2, 2);
    float g_bottom_right2  = GREEN(2, 2);

    float c_top_left2     = RAW(-2, -2);
    float c_top_right2    = RAW(2, -2);
    float c_bottom_left2  = RAW(-2, 2);
    float c_bottom_right2 = RAW(2, 2);

    float gc_top_left2     = g_top_left2     - c_top_left2;
    float gc_top_right2    = g_top_right2    - c_top_right2;
    float gc_bottom_left2  = g_bottom_left2  - c_bottom_left2;
    float gc_bottom_right2 = g_bottom_right2 - c_bottom_right2;

    // Estimate the (diagonal) gradient direction taking into account the raw noise model
    float2 variance = red_pixel ? redVariance : blueVariance;
    float rawStdDev = sqrt(variance.x + variance.y * c2_ave);
    float2 gradient = abs(read_imagef(gradientImage, imageCoordinates).xy);
    float direction = 1 - 2 * atan2pi(gradient.y, gradient.x);
    float gradient_threshold = smoothstep(rawStdDev, 4 * rawStdDev, length(gradient));
    // If the gradient is below threshold go flat
    float alpha = mix(0.5, direction, gradient_threshold);

    // Edges that are in strong highlights tend to exagerate the gradient
    float highlights_edge = 1 - smoothstep(0.25, 1.0, max(green, max(max(g_top_right2, g_bottom_left2),
                                                                     max(g_top_left2, g_bottom_right2))));

    float c2 = green - mix((gc_top_right + gc_bottom_left) / 2 + highlights_edge * (gc_top_right2 + gc_bottom_left2 - 2 * (green - c1)) / 8,
                           (gc_top_left + gc_bottom_right) / 2 + highlights_edge * (gc_top_left2 + gc_bottom_right2 - 2 * (green - c1)) / 8, alpha);

    // Limit the range of HF correction to something reasonable
    float c2max = max(max(c2_top_left, c2_top_right), max(c2_bottom_left, c2_bottom_right));
    float c2min = min(min(c2_top_left, c2_top_right), min(c2_bottom_left, c2_bottom_right));
    c2 = clamp(c2, c2min, c2max);

    float3 output = red_pixel ? (float3)(c1, green, c2) : (float3)(c2, green, c1);

    write_imagef(rgbImage, imageCoordinates, (float4)(clamp(output, 0.0, 1.0), 0));
}

kernel void interpolateRedBlue(read_only image2d_t rawImage,
                               read_only image2d_t greenImage,
                               read_only image2d_t gradientImage,
                               write_only image2d_t rgbImage,
                               int bayerPattern,
                               float2 redVariance, float2 blueVariance) {
    const int2 imageCoordinates = 2 * (int2) (get_global_id(0), get_global_id(1));

    const int2 r = bayerOffsets[bayerPattern][raw_red];
    const int2 g = bayerOffsets[bayerPattern][raw_green];
    const int2 b = bayerOffsets[bayerPattern][raw_blue];
    const int2 g2 = bayerOffsets[bayerPattern][raw_green2];

    interpolateRedBluePixel(rawImage, greenImage, gradientImage, rgbImage, redVariance, blueVariance, true, imageCoordinates + r);
    interpolateRedBluePixel(rawImage, greenImage, gradientImage, rgbImage, redVariance, blueVariance, false, imageCoordinates + b);

    write_imagef(rgbImage, imageCoordinates + g, (float4)(0, clamp(read_imagef(greenImage, imageCoordinates + g).x, 0.0, 1.0), 0, 0));
    write_imagef(rgbImage, imageCoordinates + g2, (float4)(0, clamp(read_imagef(greenImage, imageCoordinates + g2).x, 0.0, 1.0), 0, 0));
}
#undef GREEN

#define RGB(i, j) read_imagef(rgbImageIn, (int2)(imageCoordinates.x + i, imageCoordinates.y + j)).xyz

// Interpolate Red and Blue colors at Green RAW locations

void interpolateRedBlueAtGreenPixel(read_only image2d_t rgbImageIn,
                                    read_only image2d_t gradientImage,
                                    write_only image2d_t rgbImageOut,
                                    float2 redVariance, float2 blueVariance,
                                    int2 imageCoordinates) {
    float3 rgb = RGB(0, 0);

    // Green pixel locations
    float3 rgb_left     = RGB(-1, 0);
    float3 rgb_right    = RGB(1, 0);
    float3 rgb_up       = RGB(0, -1);
    float3 rgb_down     = RGB(0, 1);

    float red_ave       = (rgb_left.x + rgb_right.x + rgb_up.x + rgb_down.x) / 4;
    float blue_ave      = (rgb_left.z + rgb_right.z + rgb_up.z + rgb_down.z) / 4;

    float gred_left      = rgb_left.y  - rgb_left.x;
    float gred_right     = rgb_right.y - rgb_right.x;
    float gred_up        = rgb_up.y    - rgb_up.x;
    float gred_down      = rgb_down.y  - rgb_down.x;

    float gblue_left     = rgb_left.y  - rgb_left.z;
    float gblue_right    = rgb_right.y - rgb_right.z;
    float gblue_up       = rgb_up.y    - rgb_up.z;
    float gblue_down     = rgb_down.y  - rgb_down.z;

    float3 rgb_left3    = RGB(-3, 0);
    float3 rgb_right3   = RGB(3, 0);
    float3 rgb_up3      = RGB(0, -3);
    float3 rgb_down3    = RGB(0, 3);

    float gred_left3     = rgb_left3.y  - rgb_left3.x;
    float gred_right3    = rgb_right3.y - rgb_right3.x;
    float gred_up3       = rgb_up3.y    - rgb_up3.x;
    float gred_down3     = rgb_down3.y  - rgb_down3.x;

    float gblue_left3    = rgb_left3.y  - rgb_left3.z;
    float gblue_right3   = rgb_right3.y - rgb_right3.z;
    float gblue_up3      = rgb_up3.y    - rgb_up3.z;
    float gblue_down3    = rgb_down3.y  - rgb_down3.z;

    // Gradient direction in [0..1]
    float2 gradient = abs(read_imagef(gradientImage, imageCoordinates).xy);
    float direction = 2 * atan2pi(gradient.y, gradient.x);

    float redStdDev = sqrt(redVariance.x + redVariance.y * red_ave);
    float redGradient_threshold = smoothstep(redStdDev, 4 * redStdDev, length(gradient));

    float blueStdDev = sqrt(blueVariance.x + blueVariance.y * blue_ave);
    float blueGradient_threshold = smoothstep(blueStdDev, 4 * blueStdDev, length(gradient));

    // If the gradient is below threshold go flat
    float redAlpha = mix(0.5, direction, redGradient_threshold);
    float blueAlpha = mix(0.5, direction, blueGradient_threshold);

    // Edges that are in strong highlights tend to exagerate the gradient
    float highlights_edge = 1 - smoothstep(0.25, 1.0, max(rgb.y, max(max(rgb_right.y, rgb_left.y),
                                                                     max(rgb_down.y, rgb_up.y))));

    float red = rgb.y - mix((gred_up + gred_down) / 2 - highlights_edge * ((gred_down3 - gred_up) - (gred_down - gred_up3)) / 8,
                            (gred_left + gred_right) / 2 - highlights_edge * ((gred_right3 - gred_left) - (gred_right - gred_left3)) / 8,
                            redAlpha);

    float blue = rgb.y - mix((gblue_up + gblue_down) / 2 - highlights_edge * ((gblue_down3 - gblue_up) - (gblue_down - gblue_up3)) / 8,
                             (gblue_left + gblue_right) / 2 - highlights_edge * ((gblue_right3 - gblue_left) - (gblue_right - gblue_left3)) / 8,
                             blueAlpha);

    // Limit the range of HF correction to something reasonable
    float red_min = min(min(rgb_left.x, rgb_right.x), min(rgb_up.x, rgb_down.x));
    float red_max = max(max(rgb_left.x, rgb_right.x), max(rgb_up.x, rgb_down.x));
    rgb.x = clamp(red, red_min, red_max);

    float blue_min = min(min(rgb_left.z, rgb_right.z), min(rgb_up.z, rgb_down.z));
    float blue_max = max(max(rgb_left.z, rgb_right.z), max(rgb_up.z, rgb_down.z));
    rgb.z = clamp(blue, blue_min, blue_max);

    write_imagef(rgbImageOut, imageCoordinates, (float4)(rgb, 0));
}

kernel void interpolateRedBlueAtGreen(read_only image2d_t rgbImageIn,
                                      read_only image2d_t gradientImage,
                                      write_only image2d_t rgbImageOut,
                                      int bayerPattern, float2 redVariance, float2 blueVariance) {
    const int2 imageCoordinates = 2 * (int2) (get_global_id(0), get_global_id(1));

    const int2 g = bayerOffsets[bayerPattern][raw_green];
    const int2 g2 = bayerOffsets[bayerPattern][raw_green2];

    interpolateRedBlueAtGreenPixel(rgbImageIn, gradientImage, rgbImageOut, redVariance, blueVariance, imageCoordinates + g);
    interpolateRedBlueAtGreenPixel(rgbImageIn, gradientImage, rgbImageOut, redVariance, blueVariance, imageCoordinates + g2);
}

// Modified Malvar-He-Cutler algorithm - for reference only

kernel void malvar(read_only image2d_t rawImage, read_only image2d_t gradientImage,
                   write_only image2d_t rgbImage, int bayerPattern,
                   float2 redVariance, float2 greenVariance, float2 blueVariance) {
    const int2 imageCoordinates = (int2)(get_global_id(0), get_global_id(1));

    const int x = imageCoordinates.x;
    const int y = imageCoordinates.y;

    const int2 r = bayerOffsets[bayerPattern][raw_red];
    const int2 b = bayerOffsets[bayerPattern][raw_blue];

    const bool red_pixel = (r.x & 1) == (x & 1) && (r.y & 1) == (y & 1);
    const bool blue_pixel = (b.x & 1) == (x & 1) && (b.y & 1) == (y & 1);
    const bool red_line = (r.y & 1) == (y & 1);

    // Estimate gradient intensity and direction
    float2 gradient = abs(read_imagef(gradientImage, imageCoordinates).xy);
    // Gradient direction in [0..1]
    float direction = 2 * atan2pi(gradient.y, gradient.x);

    float green, c1, c2;
    if (red_pixel || blue_pixel) {
        // Red and Blue Pixels
        c1 = RAW(0, 0);

        float c1_left   = RAW(-2, 0);
        float c1_right  = RAW(2, 0);
        float c1_top    = RAW(0, -2);
        float c1_bottom = RAW(0, 2);

        // Edges that are in strong highlights tend to exagerate the gradient
        float highlights_edge = 1 - 0.5 * smoothstep(0.25, 1.0, max(c1, max(max(c1_left, c1_right), max(c1_top, c1_bottom))));

        float2 c2_d2 = (float2) { c1_left + c1_right, c1_top + c1_bottom } - 2 * c1;

        {
            float g_left    = RAW(-1, 0);
            float g_right   = RAW(1, 0);
            float g_up      = RAW(0, -1);
            float g_down    = RAW(0, 1);

            // Estimate gradient intensity and direction
            float g_ave = (g_up + g_down + g_left + g_right) / 4;
            float rawStdDev = sqrt(greenVariance.x + greenVariance.y * g_ave);

            // Minimum gradient threshold wrt the noise model
            float gradient_threshold = smoothstep(rawStdDev, 4 * rawStdDev, length(gradient));

            // If the gradient is below threshold interpolate against the grain
            float alpha = mix(0.5, direction, gradient_threshold);

            float2 g_lf = { 2 * (g_left + g_right), 2 * (g_up + g_down) };

            float2 g_est = g_lf - highlights_edge * c2_d2;

            green = mix(g_est.y, g_est.x, alpha) / 4;

            // Limit the range of HF correction to something reasonable
            float gmax = max(max(g_left, g_right), max(g_up, g_down));
            float gmin = min(min(g_left, g_right), min(g_up, g_down));
            green = clamp(green, gmin, gmax);
        }

        {
            float c2_top_left       = RAW(-1, -1);
            float c2_top_right      = RAW(1, -1);
            float c2_bottom_left    = RAW(-1, 1);
            float c2_bottom_right   = RAW(1, 1);

            float2 c2_lf = (float2) { (c2_top_left + c2_bottom_right) / 2, (c2_top_right + c2_bottom_left) / 2 };

            float2 dv = (float2) (fabs(c2_bottom_right - c2_top_left), fabs(c2_bottom_left - c2_top_right));
            float diagonal_direction = 2 * atan2pi(dv.y, dv.x);

            c2 = mix(c2_lf.y, c2_lf.x, diagonal_direction) - 3 * highlights_edge * mix(c2_d2.y, c2_d2.x, 0.5) / 8;

            // Limit the range of HF correction to something reasonable
            float c2max = max(max(c2_top_left, c2_bottom_right), max(c2_top_right, c2_bottom_left));
            float c2min = min(min(c2_top_left, c2_bottom_right), min(c2_top_right, c2_bottom_left));
            c2 = clamp(c2, c2min, c2max);
        }
    } else {
        // Green Pixels
        green = RAW(0, 0);

        float g_top_left     = RAW(-1, -1);
        float g_top_right    = RAW(1, -1);
        float g_bottom_left  = RAW(-1, 1);
        float g_bottom_right = RAW(1, 1);

        float g_left2   = RAW(-2, 0);
        float g_right2  = RAW(2, 0);
        float g_top2    = RAW(0, -2);
        float g_bottom2 = RAW(0, 2);

        float c1_left   = RAW(-1, 0);
        float c1_right  = RAW(1, 0);
        float c2_top    = RAW(0, -1);
        float c2_bottom = RAW(0, 1);

        float highlights_edge = 1 - 0.5 * smoothstep(0.25, 1.0, max(green, max(max(g_left2, g_right2), max(g_top2, g_bottom2))));

        c1 = (c1_left + c1_right) / 2 +
            highlights_edge * (10 * green - 2 * (g_top_left + g_top_right + g_bottom_left +
                                                 g_bottom_right + g_left2 + g_right2) +
                               g_top2 + g_bottom2) / 16;

        c2 = (c2_top + c2_bottom) / 2 +
            highlights_edge * (10 * green - 2 * (g_top_left + g_top_right + g_bottom_left +
                                                 g_bottom_right + g_top2 + g_bottom2) +
                               g_left2 + g_right2) / 16;

        // Limit the range of HF correction to something reasonable
        float c1max = max(c1_left, c1_right);
        float c1min = min(c1_left, c1_right);
        c1 = clamp(c1, c1min, c1max);

        float c2max = max(c2_top, c2_bottom);
        float c2min = min(c2_top, c2_bottom);
        c2 = clamp(c2, c2min, c2max);
    }

    float3 output = red_line ? (float3) { c1, green, c2 } : (float3) { c2, green, c1 };
    write_imagef(rgbImage, imageCoordinates, (float4)(output, 0));
}

#undef RAW

kernel void fastDebayer(read_only image2d_t rawImage, write_only image2d_t rgbImage, int bayerPattern) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const int2 r = bayerOffsets[bayerPattern][raw_red];
    const int2 g = bayerOffsets[bayerPattern][raw_green];
    const int2 b = bayerOffsets[bayerPattern][raw_blue];
    const int2 g2 = bayerOffsets[bayerPattern][raw_green2];

    float red    = read_imagef(rawImage, 2 * imageCoordinates + r).x;
    float green  = read_imagef(rawImage, 2 * imageCoordinates + g).x;
    float blue   = read_imagef(rawImage, 2 * imageCoordinates + b).x;
    float green2 = read_imagef(rawImage, 2 * imageCoordinates + g2).x;

    write_imagef(rgbImage, imageCoordinates, (float4)(red, (green + green2) / 2, blue, 0.0));
}

#define M_SQRT3_F 1.7320508f

constant float3 trans[3] = {
    {         1,          1, 1 },
    { M_SQRT3_F, -M_SQRT3_F, 0 },
    {        -1,         -1, 2 },
};
constant float3 itrans[3] = {
    { 1,  M_SQRT3_F / 2, -0.5 },
    { 1, -M_SQRT3_F / 2, -0.5 },
    { 1,              0,  1   },
};

kernel void blendHighlightsImage(read_only image2d_t inputImage, float clip, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2)(get_global_id(0), get_global_id(1));

    float3 pixel = read_imagef(inputImage, imageCoordinates).xyz;
    if (any(pixel > clip)) {
        float3 cam[2] = {pixel, min(pixel, clip)};

        float3 lab[2];
        float sum[2];
        for (int i = 0; i < 2; i++) {
            lab[i] = (float3)(dot(trans[0], cam[i]),
                              dot(trans[1], cam[i]),
                              dot(trans[2], cam[i]));
            sum[i] = dot(lab[i].yz, lab[i].yz);
        }
        float chratio = sum[0] > 0 ? sqrt(sum[1] / sum[0]) : 1;
        lab[0].yz *= chratio;

        pixel = (float3)(dot(itrans[0], lab[0]),
                         dot(itrans[1], lab[0]),
                         dot(itrans[2], lab[0])) / 3;
    }

#if LENS_SHADING
    float2 imageCenter = convert_float2(get_image_dim(inputImage) / 2);
    float distance_from_center = length(convert_float2(imageCoordinates) - imageCenter) / length(imageCenter);
    float lens_shading = 1 + LENS_SHADING_GAIN * distance_from_center * distance_from_center;
    pixel *= lens_shading;
#endif

    write_imagef(outputImage, imageCoordinates, (float4)(pixel, 0.0));
}

/// ---- Median Filter 3x3 ----

#define s(a, b)                         \
  ({ typedef __typeof__ (a) type_of_a;  \
     type_of_a temp = a;                \
     a = min(a, b);                     \
     b = max(temp, b); })

#define minMax6(a0,a1,a2,a3,a4,a5) s(a0,a1);s(a2,a3);s(a4,a5);s(a0,a2);s(a1,a3);s(a0,a4);s(a3,a5);
#define minMax5(a0,a1,a2,a3,a4) s(a0,a1);s(a2,a3);s(a0,a2);s(a1,a3);s(a0,a4);s(a3,a4);
#define minMax4(a0,a1,a2,a3) s(a0,a1);s(a2,a3);s(a0,a2);s(a1,a3);
#define minMax3(a0,a1,a2) s(a0,a1);s(a0,a2);s(a1,a2);

#define fast_median3x3(inputImage, imageCoordinates)               \
({                                                                 \
    medianPixelType a0, a1, a2, a3, a4, a5;                        \
                                                                   \
    a0 = readImage(inputImage, imageCoordinates + (int2)(0, -1));  \
    a1 = readImage(inputImage, imageCoordinates + (int2)(1, -1));  \
    a2 = readImage(inputImage, imageCoordinates + (int2)(0, 0));   \
    a3 = readImage(inputImage, imageCoordinates + (int2)(1, 0));   \
    a4 = readImage(inputImage, imageCoordinates + (int2)(0, 1));   \
    a5 = readImage(inputImage, imageCoordinates + (int2)(1, 1));   \
    minMax6(a0, a1, a2, a3, a4, a5);                               \
    a0 = readImage(inputImage, imageCoordinates + (int2)(-1, 1));  \
    minMax5(a0, a1, a2, a3, a4);                                   \
    a0 = readImage(inputImage, imageCoordinates + (int2)(-1, 0));  \
    minMax4(a0, a1, a2, a3);                                       \
    a0 = readImage(inputImage, imageCoordinates + (int2)(-1, -1)); \
    minMax3(a0, a1, a2);                                           \
    a1;                                                            \
})

#define readImage(image, pos)  read_imageh(image, pos).xy;

kernel void medianFilterImage3x3x2(read_only image2d_t inputImage, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    typedef half2 medianPixelType;

    half2 median = fast_median3x3(inputImage, imageCoordinates);

    write_imageh(denoisedImage, imageCoordinates, (half4) (median, 0, 0));
}

#undef readImage

#define readImage(image, pos)  read_imageh(image, pos).xyz;

kernel void medianFilterImage3x3x3(read_only image2d_t inputImage, write_only image2d_t filteredImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    typedef half3 medianPixelType;

    half3 median = fast_median3x3(inputImage, imageCoordinates);

    write_imageh(filteredImage, imageCoordinates, (half4) (median, 0));
}

#undef readImage

#define readImage(image, pos)  read_imageh(image, pos);

kernel void medianFilterImage3x3x4(read_only image2d_t inputImage, write_only image2d_t filteredImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    typedef half4 medianPixelType;

    half4 median = fast_median3x3(inputImage, imageCoordinates);

    write_imageh(filteredImage, imageCoordinates, median);
}

#undef readImage

#define readImage(image, pos)                              \
    ({                                                     \
        half3 p = read_imageh(image, pos).xyz;             \
                                                           \
        half v = p.x;                                      \
        secMax = v <= firstMax && v > secMax ? v : secMax; \
        secMax = v > firstMax ? firstMax : secMax;         \
        firstMax = v > firstMax ? v : firstMax;            \
                                                           \
        secMin = v >= firstMin && v < secMin ? v : secMin; \
        secMin = v < firstMin ? firstMin : secMin;         \
        firstMin = v < firstMin ? v : firstMin;            \
                                                           \
        if (all(pos == imageCoordinates)) {                \
            sample = v;                                    \
        }                                                  \
                                                           \
        p.yz;                                              \
    })

kernel void despeckleLumaMedianChromaImage(read_only image2d_t inputImage, float3 var_a, float3 var_b,
                                           write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    half sample = 0, firstMax = 0, secMax = 0;
    half firstMin = (float) 0xffff, secMin = (float) 0xffff;

    // Median filtering of the chroma
    typedef half2 medianPixelType;
    half2 median = fast_median3x3(inputImage, imageCoordinates);

    half sigma = sqrt(var_a.x + var_b.x * sample);
    half minVal = mix(secMin, firstMin, smoothstep(sigma, 4 * sigma, secMin - firstMin));
    half maxVal = mix(secMax, firstMax, smoothstep(sigma, 4 * sigma, firstMax - secMax));

    sample = clamp(sample, minVal, maxVal);

    write_imageh(denoisedImage, imageCoordinates, (half4) (sample, median, 0));
}

#undef readImage

#undef minMax6
#undef minMax5
#undef minMax4
#undef minMax3
#undef s

// ---- Despeckle ----

half despeckle_3x3(image2d_t inputImage, float inputLuma, float var_a, float var_b, int2 imageCoordinates) {
    half sample = 0, firstMax = 0, secMax = 0;
    half firstMin = (half) 100, secMin = (half) 100;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            half v = read_imageh(inputImage, imageCoordinates + (int2)(x, y)).x;

            secMax = v <= firstMax && v > secMax ? v : secMax;
            secMax = v > firstMax ? firstMax : secMax;
            firstMax = v > firstMax ? v : firstMax;

            secMin = v >= firstMin && v < secMin ? v : secMin;
            secMin = v < firstMin ? firstMin : secMin;
            firstMin = v < firstMin ? v : firstMin;

            if (x == 0 && y == 0) {
                sample = v;
            }
        }
    }

    half sigma = sqrt(var_a + var_b * inputLuma);
    half minVal = mix(secMin, firstMin, smoothstep(sigma, 4 * sigma, secMin - firstMin));
    half maxVal = mix(secMax, firstMax, smoothstep(sigma, 4 * sigma, firstMax - secMax));

    return clamp(sample, minVal, maxVal);
}

kernel void despeckleImage(read_only image2d_t inputImage, float3 var_a, float3 var_b, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    half3 inputPixel = read_imageh(inputImage, imageCoordinates).xyz;

    half denoisedLuma = despeckle_3x3(inputImage, inputPixel.x, var_a.x, var_b.x, imageCoordinates);

    write_imageh(denoisedImage, imageCoordinates, (half4) (denoisedLuma, inputPixel.yz, 0.0));
}

half4 despeckle_3x3x4(image2d_t inputImage, float4 rawVariance, int2 imageCoordinates) {
    half4 sample = 0, firstMax = 0, secondMax = 0;
    half4 firstMin = (half) HALF_MAX, secondMin = (half) HALF_MAX;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            half4 v = read_imageh(inputImage, imageCoordinates + (int2)(x, y));

            secondMax = v >= firstMax ? firstMax : max(v, secondMax);
            firstMax = max(v, firstMax);

            secondMin = v <= firstMin ? firstMin : min(v, secondMin);
            firstMin = min(v, firstMin);

            if (x == 0 && y == 0) {
                sample = v;
            }
        }
    }

    half4 sigma = sqrt(convert_half4(rawVariance) * sample);
    half4 minVal = mix(secondMin, firstMin, smoothstep(2 * sigma, 8 * sigma, secondMin - firstMin));
    half4 maxVal = mix(secondMax, firstMax, smoothstep(sigma, 4 * sigma, firstMax - secondMax));
    return clamp(sample, minVal, maxVal);
}

half4 despeckle_3x3x4_strong(image2d_t inputImage, float4 rawVariance, int2 imageCoordinates) {
    half4 sample = 0, firstMax = 0, secondMax = 0, thirdMax = 0;
    half4 firstMin = (half) HALF_MAX, secondMin = (half) HALF_MAX, thirdMin = (half) HALF_MAX;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            half4 v = read_imageh(inputImage, imageCoordinates + (int2)(x, y));

            thirdMax = v >= secondMax ? secondMax : max(v, thirdMax);
            secondMax = v >= firstMax ? firstMax : max(v, secondMax);
            firstMax = max(v, firstMax);

            thirdMin = v <= secondMin ? secondMin : min(v, thirdMin);
            secondMin = v <= firstMin ? firstMin : min(v, secondMin);
            firstMin = min(v, firstMin);

            if (x == 0 && y == 0) {
                sample = v;
            }
        }
    }

    half4 sigma = sqrt(convert_half4(rawVariance) * sample);
    half4 minVal = mix(thirdMin, firstMin, smoothstep(2 * sigma, 8 * sigma, thirdMin - firstMin));
    half4 maxVal = mix(thirdMax, firstMax, smoothstep(sigma, 4 * sigma, firstMax - thirdMax));
    return clamp(sample, minVal, maxVal);
}

kernel void despeckleRawRGBAImage(read_only image2d_t inputImage, float4 rawVariance, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    half4 despeckledPixel = despeckle_3x3x4(inputImage, rawVariance, imageCoordinates);

    write_imageh(denoisedImage, imageCoordinates, despeckledPixel);
}

/*
 * 5 x 5 Fast Median Filter Implementation for Chroma Antialiasing
 */

#define s(a, b)                         \
  ({ typedef __typeof__ (a) type_of_a;  \
     type_of_a temp = a;                \
     a = min(a, b);                     \
     b = max(temp, b); })

#define minMax14(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13) s(a0,a1);s(a2,a3);s(a4,a5);s(a6,a7);s(a8,a9);s(a10,a11);s(a12,a13);s(a0,a2);s(a4,a6);s(a8,a10);s(a1,a3);s(a5,a7);s(a9,a11);s(a0,a4);s(a8,a12);s(a3,a7);s(a11,a13);s(a0,a8);s(a7,a13);
#define minMax13(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12) s(a0,a1);s(a2,a3);s(a4,a5);s(a6,a7);s(a8,a9);s(a10,a11);s(a0,a2);s(a4,a6);s(a8,a10);s(a1,a3);s(a5,a7);s(a9,a11);s(a0,a4);s(a8,a12);s(a3,a7);s(a11,a12);s(a0,a8);s(a7,a12);
#define minMax12(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11) s(a0,a1);s(a2,a3);s(a4,a5);s(a6,a7);s(a8,a9);s(a10,a11);s(a0,a2);s(a4,a6);s(a8,a10);s(a1,a3);s(a5,a7);s(a9,a11);s(a0,a4);s(a3,a7);s(a0,a8);s(a7,a11);
#define minMax11(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10) s(a0,a1);s(a2,a3);s(a4,a5);s(a6,a7);s(a8,a9);s(a0,a2);s(a4,a6);s(a8,a10);s(a1,a3);s(a5,a7);s(a9,a10);s(a0,a4);s(a3,a7);s(a0,a8);s(a7,a10);
#define minMax10(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9) s(a0,a1);s(a2,a3);s(a4,a5);s(a6,a7);s(a8,a9);s(a0,a2);s(a4,a6);s(a1,a3);s(a5,a7);s(a0,a4);s(a3,a7);s(a0,a8);s(a7,a9);
#define minMax9(a0,a1,a2,a3,a4,a5,a6,a7,a8) s(a0,a1);s(a2,a3);s(a4,a5);s(a6,a7);s(a0,a2);s(a4,a6);s(a1,a3);s(a5,a7);s(a0,a4);s(a3,a7);s(a0,a8);s(a7,a8);
#define minMax8(a0,a1,a2,a3,a4,a5,a6,a7) s(a0,a1);s(a2,a3);s(a4,a5);s(a6,a7);s(a0,a2);s(a4,a6);s(a1,a3);s(a5,a7);s(a0,a4);s(a3,a7);
#define minMax7(a0,a1,a2,a3,a4,a5,a6) s(a0,a1);s(a2,a3);s(a4,a5);s(a0,a2);s(a4,a6);s(a1,a3);s(a5,a6);s(a0,a4);s(a3,a6);
#define minMax6(a0,a1,a2,a3,a4,a5) s(a0,a1);s(a2,a3);s(a4,a5);s(a0,a2);s(a1,a3);s(a0,a4);s(a3,a5);
#define minMax5(a0,a1,a2,a3,a4) s(a0,a1);s(a2,a3);s(a0,a2);s(a1,a3);s(a0,a4);s(a3,a4);
#define minMax4(a0,a1,a2,a3) s(a0,a1);s(a2,a3);s(a0,a2);s(a1,a3);
#define minMax3(a0,a1,a2) s(a0,a1);s(a0,a2);s(a1,a2);

#define fast_median5x5(inputImage, imageCoordinates)                                \
    ({                                                                              \
        medianPixelType a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13; \
                                                                                    \
        a0 = readImage(inputImage, imageCoordinates + (int2)(-1, -2));              \
        a1 = readImage(inputImage, imageCoordinates + (int2)(0, -2));               \
        a2 = readImage(inputImage, imageCoordinates + (int2)(1, -2));               \
        a3 = readImage(inputImage, imageCoordinates + (int2)(2, -2));               \
        a4 = readImage(inputImage, imageCoordinates + (int2)(-1, -1));              \
        a5 = readImage(inputImage, imageCoordinates + (int2)(0, -1));               \
        a6 = readImage(inputImage, imageCoordinates + (int2)(1, -1));               \
        a7 = readImage(inputImage, imageCoordinates + (int2)(2, -1));               \
        a8 = readImage(inputImage, imageCoordinates + (int2)(-1, 0));               \
        a9 = readImage(inputImage, imageCoordinates + (int2)(0, 0));                \
        a10 = readImage(inputImage, imageCoordinates + (int2)(1, 0));               \
        a11 = readImage(inputImage, imageCoordinates + (int2)(2, 0));               \
        a12 = readImage(inputImage, imageCoordinates + (int2)(-1, 1));              \
        a13 = readImage(inputImage, imageCoordinates + (int2)(0, 1));               \
        minMax14(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);       \
        a0 = readImage(inputImage, imageCoordinates + (int2)(1, 1));                \
        minMax13(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);            \
        a0 = readImage(inputImage, imageCoordinates + (int2)(2, 1));                \
        minMax12(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);                 \
        a0 = readImage(inputImage, imageCoordinates + (int2)(-1, 2));               \
        minMax11(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);                      \
        a0 = readImage(inputImage, imageCoordinates + (int2)(0, 2));                \
        minMax10(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);                           \
        a0 = readImage(inputImage, imageCoordinates + (int2)(1, 2));                \
        minMax9(a0, a1, a2, a3, a4, a5, a6, a7, a8);                                \
        a0 = readImage(inputImage, imageCoordinates + (int2)(2, 2));                \
        minMax8(a0, a1, a2, a3, a4, a5, a6, a7);                                    \
        a0 = readImage(inputImage, imageCoordinates + (int2)(-2, 2));               \
        minMax7(a0, a1, a2, a3, a4, a5, a6);                                        \
        a0 = readImage(inputImage, imageCoordinates + (int2)(-2, 1));               \
        minMax6(a0, a1, a2, a3, a4, a5);                                            \
        a0 = readImage(inputImage, imageCoordinates + (int2)(-2, 0));               \
        minMax5(a0, a1, a2, a3, a4);                                                \
        a0 = readImage(inputImage, imageCoordinates + (int2)(-2, -1));              \
        minMax4(a0, a1, a2, a3);                                                    \
        a0 = readImage(inputImage, imageCoordinates + (int2)(-2, -2));              \
        minMax3(a0, a1, a2);                                                        \
        a1;                                                                         \
    })


#define readImage(image, pos)  read_imageh(image, pos).xyz;

kernel void medianFilterImage5x5x3(read_only image2d_t inputImage, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    typedef half3 medianPixelType;

    half3 median = fast_median5x5(inputImage, imageCoordinates);

    write_imageh(denoisedImage, imageCoordinates, (half4) (median, 0));
}

#undef readImage

#define readImage(image, pos)  read_imageh(image, pos);

kernel void medianFilterImage5x5x4(read_only image2d_t inputImage, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    typedef half4 medianPixelType;

    half4 median = fast_median5x5(inputImage, imageCoordinates);

    write_imageh(denoisedImage, imageCoordinates, median);
}

#undef readImage


// ---- False Colors Removal ----

/*
 See: "False colors removal on the YCr-Cb color space", V. Tomaselli, M. Guarnera, G. Messina
      https://www.researchgate.net/publication/221159269_False_colors_removal_on_the_YCr-Cb_color_space
 */

// Read image elements for median filter and collect pixel statistics

#define readImage(image, pos)                         \
    ({                                                \
        half3 p = read_imageh(image, pos).xyz;        \
        max = p > max ? p : max;                      \
        min = p < min ? p : min;                      \
        half W = 1 / (1 + p.x - inputPixel.x);        \
        crossCorrelation += p.yz * W;                 \
        centerCorrelation += W > 0.6;                 \
        sumW += W;                                    \
        p.yz;                                         \
    })

// False Colors Removal kernel, see cited paper for algorithm details

kernel void falseColorsRemovalImage(read_only image2d_t inputImage, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    half3 inputPixel = read_imageh(inputImage, imageCoordinates).xyz;

    // Data type of the median filter
    typedef half2 medianPixelType;

    // Region statistics
    half3 max = -100;
    half3 min =  100;
    half2 crossCorrelation = 0;
    half sumW = 0;
    int centerCorrelation = 0;

    // Compute the median filter of the chroma and extract the region's statistics
    half2 chromaMedian = fast_median5x5(inputImage, imageCoordinates);

    // Correction Factor from edge strenght estimation
    half3 D = max - min;
    half cf = D.x < D.y && D.x < D.x ? D.x : max(D.x, max(D.y, D.z));
    cf = exp(-312.5 * cf * cf);

    // Inter-channel correlation penalty factor
    crossCorrelation = centerCorrelation > 1 ? crossCorrelation / (inputPixel.yz * sumW) : 1;

    // Mix the chroma median with the original signal according
    half2 chroma = mix(chromaMedian, inputPixel.yz, min(cf + crossCorrelation * crossCorrelation, 1));

    write_imageh(denoisedImage, imageCoordinates, (half4) (inputPixel.x, chroma, 0));
}
#undef readImage

#undef s
#undef minMax14
#undef minMax13
#undef minMax12
#undef minMax11
#undef minMax10
#undef minMax9
#undef minMax8
#undef minMax7
#undef minMax6
#undef minMax5
#undef minMax4
#undef minMax3

/// ---- Image Denoising ----

typedef struct {
    float3 m[3];
} Matrix3x3;

kernel void transformImage(read_only image2d_t inputImage, write_only image2d_t outputImage, Matrix3x3 transform) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    float3 inputValue = read_imagef(inputImage, imageCoordinates).xyz;
    float3 outputPixel = (float3) (dot(transform.m[0], inputValue), dot(transform.m[1], inputValue), dot(transform.m[2], inputValue));
    write_imagef(outputImage, imageCoordinates, (float4) (outputPixel, 0.0));
}

half tunnel(half x, half y, half angle, half sigma) {
    half a = x * cos(angle) + y * sin(angle);
    return exp(-(a * a) / sigma);
}

kernel void denoiseImage(read_only image2d_t inputImage,
                         read_only image2d_t gradientImage,
                         float3 var_a, float3 var_b, float3 thresholdMultipliers,
                         float chromaBoost, float gradientBoost, float gradientThreshold,
                         write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const half3 inputYCC = read_imageh(inputImage, imageCoordinates).xyz;

    half3 sigma = convert_half3(sqrt(var_a + var_b * inputYCC.x));
    half3 diffMultiplier = 1 / (convert_half3(thresholdMultipliers) * sigma);

    half2 gradient = read_imageh(gradientImage, imageCoordinates).xy;
    half angle = atan2(gradient.y, gradient.x);
    half magnitude = length(gradient);
    half edge = smoothstep(4, 16, gradientThreshold * magnitude / sigma.x);

    const int size = gradientBoost > 0 ? 4 : 2;

    half3 filtered_pixel = 0;
    half3 kernel_norm = 0;
    for (int y = -size; y <= size; y++) {
        for (int x = -size; x <= size; x++) {
            half3 inputSampleYCC = read_imageh(inputImage, imageCoordinates + (int2)(x, y)).xyz;
            half2 gradientSample = read_imageh(gradientImage, imageCoordinates + (int2)(x, y)).xy;

            half3 inputDiff = (inputSampleYCC - inputYCC) * diffMultiplier;
            half2 gradientDiff = (gradientSample - gradient) / sigma.x;

            half directionWeight = mix(1, tunnel(x, y, angle, (half) 0.25), edge);
            half gradientWeight = 1 - smoothstep(2, 8, length(gradientDiff));

            half lumaWeight = 1 - step(1 + (half) gradientBoost * edge, abs(inputDiff.x));
            half chromaWeight = 1 - step((half) chromaBoost, length(inputDiff));

            half3 sampleWeight = (half3) (directionWeight * gradientWeight * lumaWeight, chromaWeight, chromaWeight);

            filtered_pixel += sampleWeight * inputSampleYCC;
            kernel_norm += sampleWeight;
        }
    }
    half3 denoisedPixel = filtered_pixel / kernel_norm;

    write_imageh(denoisedImage, imageCoordinates, (half4) (denoisedPixel, magnitude));
}

typedef struct transform {
    float matrix[3][3];
} transform;

float2 applyHomography(const transform* homography, float2 p) {
    float u = homography->matrix[0][0] * p.x + homography->matrix[0][1] * p.y + homography->matrix[0][2];
    float v = homography->matrix[1][0] * p.x + homography->matrix[1][1] * p.y + homography->matrix[1][2];
    float w = homography->matrix[2][0] * p.x + homography->matrix[2][1] * p.y + homography->matrix[2][2];
    return (float2) (u / w, v / w);
}

kernel void fuseFrames(read_only image2d_t referenceImage,
                       read_only image2d_t gradientImage,
                       read_only image2d_t inputImage,
                       read_only image2d_t fusedInputImage,
                       const transform homography,
                       sampler_t linear_sampler,
                       float3 var_a, float3 var_b, int fusedFrames,
                       write_only image2d_t fusedOutputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 input_norm = 1.0 / convert_float2(get_image_dim(fusedOutputImage));

    half3 referencePixel = read_imageh(referenceImage, imageCoordinates).xyz;
    half3 sigma = convert_half3(sqrt(var_a + var_b * referencePixel.x));

    half2 gradient = read_imageh(gradientImage, imageCoordinates).xy;
    half angle = atan2(gradient.y, gradient.x);
    half magnitude = length(gradient);
    half edge = smoothstep(1, 4, magnitude / sigma.x);

    half outWeight = 0;
    half3 outSum = 0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            half directionWeight = mix(1, tunnel(x, y, angle, (half) 0.25), edge);

            float2 pt = applyHomography(&homography, (float2) (imageCoordinates.x + x, imageCoordinates.y + y));
            half3 newPixel = read_imageh(inputImage, linear_sampler, (pt + 0.5) * input_norm).xyz;

            half weight = 1 - smoothstep((half) 0.5, (half) 2.0, (1 - (half) 0.75 * edge) * length((referencePixel - newPixel) / sigma));
            half3 outputPixel = directionWeight * weight * newPixel;

            outWeight += directionWeight * weight;
            outSum += outputPixel;
        }
    }
    outSum /= max(outWeight, 1);

    half weight = min(outWeight, 1.0);

    half3 fusedPixel = read_imageh(fusedInputImage, imageCoordinates).xyz;
    half3 outputPixel = (weight * outSum + (fusedFrames + 1 - weight) * fusedPixel) / (fusedFrames + 1);

    write_imageh(fusedOutputImage, imageCoordinates, (half4) (outputPixel, 0));
}

float3 denoiseLumaChromaGuided(float3 var_a, float3 var_b, image2d_t inputImage, int2 imageCoordinates) {
    const float3 input = read_imagef(inputImage, imageCoordinates).xyz;

    float3 eps = var_a + var_b * input.x;

    const int radius = 2;
    const float norm = 1.0 / ((2 * radius + 1) * (2 * radius + 1));

    float3 mean_I = 0;
    float mean_I_rr = 0;
    float mean_I_rg = 0;
    float mean_I_rb = 0;
    float mean_I_gg = 0;
    float mean_I_gb = 0;
    float mean_I_bb = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float3 sample = read_imagef(inputImage, imageCoordinates + (int2)(x, y)).xyz;
            mean_I += sample;
            mean_I_rr += sample.x * sample.x;
            mean_I_rg += sample.x * sample.y;
            mean_I_rb += sample.x * sample.z;
            mean_I_gg += sample.y * sample.y;
            mean_I_gb += sample.y * sample.z;
            mean_I_bb += sample.z * sample.z;
        }
    }
    mean_I *= norm;
    mean_I_rr *= norm;
    mean_I_rg *= norm;
    mean_I_rb *= norm;
    mean_I_gg *= norm;
    mean_I_gb *= norm;
    mean_I_bb *= norm;

    float var_I_rr = mean_I_rr - mean_I.x * mean_I.x;
    float var_I_rg = mean_I_rg - mean_I.x * mean_I.y;
    float var_I_rb = mean_I_rb - mean_I.x * mean_I.z;
    float var_I_gg = mean_I_gg - mean_I.y * mean_I.y;
    float var_I_gb = mean_I_gb - mean_I.y * mean_I.z;
    float var_I_bb = mean_I_bb - mean_I.z * mean_I.z;

    float var_I_rr_eps = var_I_rr + eps.x;
    float var_I_gg_eps = var_I_gg + eps.y;
    float var_I_bb_eps = var_I_bb + eps.z;

    float invrr = var_I_gg_eps * var_I_bb_eps - var_I_gb     * var_I_gb;
    float invrg = var_I_gb     * var_I_rb     - var_I_rg     * var_I_bb_eps;
    float invrb = var_I_rg     * var_I_gb     - var_I_gg_eps * var_I_rb;
    float invgg = var_I_rr_eps * var_I_bb_eps - var_I_rb     * var_I_rb;
    float invgb = var_I_rb     * var_I_rg     - var_I_rr_eps * var_I_gb;
    float invbb = var_I_rr_eps * var_I_gg_eps - var_I_rg     * var_I_rg;

    float invCovDet = 1 / (invrr * var_I_rr_eps + invrg * var_I_rg + invrb * var_I_rb);

    invrr *= invCovDet;
    invrg *= invCovDet;
    invrb *= invCovDet;
    invgg *= invCovDet;
    invgb *= invCovDet;
    invbb *= invCovDet;

    // Compute the result

    // covariance of (I, p) in each local patch.
    float3 cov_Ip_r = (float3) (var_I_rr, var_I_rg, var_I_rb);
    float3 cov_Ip_g = (float3) (var_I_rg, var_I_gg, var_I_gb);
    float3 cov_Ip_b = (float3) (var_I_rb, var_I_gb, var_I_bb);

    float3 a_r = invrr * cov_Ip_r + invrg * cov_Ip_g + invrb * cov_Ip_b;
    float3 a_g = invrg * cov_Ip_r + invgg * cov_Ip_g + invgb * cov_Ip_b;
    float3 a_b = invrb * cov_Ip_r + invgb * cov_Ip_g + invbb * cov_Ip_b;

    float3 b = mean_I - a_r * mean_I.x - a_g * mean_I.y - a_b * mean_I.z; // Eqn. (15) in the paper;

    return a_r * input.x + a_g * input.y + a_b * input.z + b;
}

kernel void denoiseImageGuided(read_only image2d_t inputImage, float3 var_a, float3 var_b, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float3 denoisedPixel = denoiseLumaChromaGuided(var_a, var_b, inputImage, imageCoordinates);

    write_imagef(denoisedImage, imageCoordinates, (float4) (denoisedPixel, 0.0));
}

kernel void downsampleImageXYZ(read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 output_pos = (int2) (get_global_id(0), get_global_id(1));
    const float2 input_norm = 1.0 / convert_float2(get_image_dim(outputImage));
    const float2 input_pos = (convert_float2(output_pos) + 0.5) * input_norm;

    // Sub-Pixel Sampling Location
    const float2 s = 0.5 * input_norm;
    float3 outputPixel = read_imagef(inputImage, linear_sampler, input_pos + (float2)(-s.x, -s.y)).xyz;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)( s.x, -s.y)).xyz;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)(-s.x,  s.y)).xyz;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)( s.x,  s.y)).xyz;
    write_imagef(outputImage, output_pos, (float4) (0.25 * outputPixel, 0));
}

kernel void downsampleImageXY(read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 output_pos = (int2) (get_global_id(0), get_global_id(1));
    const float2 input_norm = 1.0 / convert_float2(get_image_dim(outputImage));
    const float2 input_pos = (convert_float2(output_pos) + 0.5) * input_norm;

    // Sub-Pixel Sampling Location
    const float2 s = 0.5 * input_norm;
    float2 outputPixel = read_imagef(inputImage, linear_sampler, input_pos + (float2)(-s.x, -s.y)).xy;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)( s.x, -s.y)).xy;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)(-s.x,  s.y)).xy;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)( s.x,  s.y)).xy;
    write_imagef(outputImage, output_pos, (float4) (0.25 * outputPixel, 0, 0));
}

float3 applyTransform(float3 value, Matrix3x3 *transform) {
    return (float3) (dot(transform->m[0], value), dot(transform->m[1], value), dot(transform->m[2], value));
}

kernel void subtractNoiseImage(read_only image2d_t inputImage, read_only image2d_t inputImage1,
                               read_only image2d_t inputImageDenoised1, read_only image2d_t gradientImage,
                               float luma_weight, float sharpening, float2 nlf,
                               write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 output_pos = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));
    const float2 input_pos = (convert_float2(output_pos) + 0.5) * inputNorm;

    float4 inputPixel = read_imagef(inputImage, output_pos);

    float3 inputPixel1 = read_imagef(inputImage1, linear_sampler, input_pos).xyz;
    float3 inputPixelDenoised1 = read_imagef(inputImageDenoised1, linear_sampler, input_pos).xyz;

    float3 denoisedPixel = inputPixel.xyz - (float3)(luma_weight, 1, 1) * (inputPixel1 - inputPixelDenoised1);

    if (sharpening > 1.0) {
        float2 gradient = read_imagef(gradientImage, output_pos).xy;
        float sigma = sqrt(nlf.x + nlf.y * inputPixelDenoised1.x);
        float detail = smoothstep(sigma, 4 * sigma, length(gradient))
                       * (1.0 - smoothstep(0.95, 1.0, denoisedPixel.x))          // Highlights ringing protection
                       * (0.6 + 0.4 * smoothstep(0.0, 0.1, denoisedPixel.x));    // Shadows ringing protection
        sharpening = 1 + (sharpening - 1) * detail;
    }

    // Sharpen all components
    denoisedPixel = mix(inputPixelDenoised1, denoisedPixel, sharpening);
    denoisedPixel.x = max(denoisedPixel.x, 0.0);

    write_imagef(outputImage, output_pos, (float4) (denoisedPixel, inputPixel.w));
}

kernel void subtractNoiseFusedImage(read_only image2d_t inputImage, read_only image2d_t inputImage1,
                                    read_only image2d_t inputImageDenoised1, write_only image2d_t outputImage,
                                    sampler_t linear_sampler) {
    const int2 output_pos = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));
    const float2 input_pos = (convert_float2(output_pos) + 0.5) * inputNorm;

    float4 inputPixel = read_imagef(inputImage, output_pos);

    float3 inputPixel1 = read_imagef(inputImage1, linear_sampler, input_pos).xyz;
    float3 inputPixelDenoised1 = read_imagef(inputImageDenoised1, linear_sampler, input_pos).xyz;

    float3 denoisedPixel = inputPixel.xyz - (inputPixel1 - inputPixelDenoised1);

    write_imagef(outputImage, output_pos, (float4) (denoisedPixel, inputPixel.w));
}

kernel void bayerToRawRGBA(read_only image2d_t rawImage, write_only image2d_t rgbaImage, int bayerPattern) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const int2 r = bayerOffsets[bayerPattern][raw_red];
    const int2 g = bayerOffsets[bayerPattern][raw_green];
    const int2 b = bayerOffsets[bayerPattern][raw_blue];
    const int2 g2 = bayerOffsets[bayerPattern][raw_green2];

    float red    = read_imagef(rawImage, 2 * imageCoordinates + r).x;
    float green  = read_imagef(rawImage, 2 * imageCoordinates + g).x;
    float blue   = read_imagef(rawImage, 2 * imageCoordinates + b).x;
    float green2 = read_imagef(rawImage, 2 * imageCoordinates + g2).x;

    write_imagef(rgbaImage, imageCoordinates, (float4)(red, green, blue, green2));
}

kernel void rawRGBAToBayer(read_only image2d_t rgbaImage, write_only image2d_t rawImage, int bayerPattern) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float4 rgba = read_imagef(rgbaImage, imageCoordinates);

    const int2 r = bayerOffsets[bayerPattern][raw_red];
    const int2 g = bayerOffsets[bayerPattern][raw_green];
    const int2 b = bayerOffsets[bayerPattern][raw_blue];
    const int2 g2 = bayerOffsets[bayerPattern][raw_green2];

    write_imagef(rawImage, 2 * imageCoordinates + r, rgba.x);
    write_imagef(rawImage, 2 * imageCoordinates + g, rgba.y);
    write_imagef(rawImage, 2 * imageCoordinates + b, rgba.z);
    write_imagef(rawImage, 2 * imageCoordinates + g2, rgba.w);
}

void __attribute__((overloadable)) loadPatch(read_only image2d_t inputImage, const int2 imageCoordinates, float4 patch[9]) {
    for (int y = -1, i = 0; y <= 1; y++) {
        for (int x = -1; x <= 1; x++, i++) {
            patch[i] = read_imagef(inputImage, imageCoordinates + (int2)(x, y));
        }
    }
}

float4 __attribute__((overloadable)) diffPatch(read_only image2d_t inputImage, const int2 imageCoordinates, float4 patch[9]) {
    float4 diffSum = 0;
    for (int y = -1, i = 0; y <= 1; y++) {
        for (int x = -1; x <= 1; x++, i++) {
            float4 sample = read_imagef(inputImage, imageCoordinates + (int2)(x, y));
            float4 diff = sample - patch[i];
            diffSum += diff * diff;
        }
    }
    return sqrt(diffSum / 9);
}

float4 denoiseRawRGBAPatch(float4 rawVariance, image2d_t inputImage, int2 imageCoordinates) {
    float4 patch[9];
    loadPatch(inputImage, imageCoordinates, patch);

    const float4 input = read_imagef(inputImage, imageCoordinates);

    float4 sigma = sqrt(rawVariance * input);

    float4 filtered_pixel = 0;
    float4 kernel_norm = 0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float4 diff = diffPatch(inputImage, imageCoordinates + (int2)(x, y), patch) / sigma;
            float4 sampleWeight = gaussianBlur5x5[y + 2][x + 2] * (1 - step(1.0, diff));
            float4 inputSample = read_imagef(inputImage, imageCoordinates + (int2)(x, y));

            filtered_pixel += sampleWeight * inputSample;
            kernel_norm += sampleWeight;
        }
    }
    return filtered_pixel / kernel_norm;
}

float4 denoiseRawRGBA(float4 rawVariance, image2d_t inputImage, int2 imageCoordinates) {
    const float4 input = read_imagef(inputImage, imageCoordinates);

    float4 sigma = sqrt(rawVariance * input);

    float4 filtered_pixel = 0;
    float4 kernel_norm = 0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float4 inputSample = read_imagef(inputImage, imageCoordinates + (int2)(x, y));

            float4 inputDiff = (inputSample - input) / sigma;
            float4 sampleWeight = 1 - step(1.5, length(inputDiff));

            filtered_pixel += sampleWeight * inputSample;
            kernel_norm += sampleWeight;
        }
    }
    return filtered_pixel / kernel_norm;
}

kernel void denoiseRawRGBAImage(read_only image2d_t inputImage, float4 rawVariance, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float4 denoisedPixel = denoiseRawRGBA(rawVariance, inputImage, imageCoordinates);

    write_imagef(denoisedImage, imageCoordinates, denoisedPixel);
}

// Local Tone Mapping - guideImage can be a downsampled version of inputImage

kernel void GuidedFilterABImage(read_only image2d_t guideImage, write_only image2d_t abImage, float eps, sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(guideImage));
    const float2 pos = convert_float2(imageCoordinates) * inputNorm;

    float sum = 0;
    float sumSq = 0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float sample = read_imagef(guideImage, linear_sampler, pos + ((float2)(x, y) + 0.5f) * inputNorm).x;
            sum += sample;
            sumSq += sample * sample;
        }
    }
    float mean = sum / 25;
    float var = (sumSq - sum * sum / 25) / 25;

    float a = var / (var + eps);
    float b = mean * (1 - a);

    write_imagef(abImage, imageCoordinates, (float4)(a, b, 0, 0));
}

kernel void BoxFilterGFImage(read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(inputImage));
    const float2 pos = convert_float2(imageCoordinates) * inputNorm;

    float2 meanAB = 0;
    for (int i = 0; i < 9; i++) {
        constant ConvolutionParameters* cp = &boxFilter5x5[i];
        meanAB += cp->weight * read_imagef(inputImage, linear_sampler, pos + (cp->offset + 0.5f) * inputNorm).xy;
    }

    write_imagef(outputImage, imageCoordinates, (float4)(meanAB, 0, 0));
}

float computeLtmMultiplier(float3 input, float2 gfAb, float eps, float shadows,
                           float highlights, float detail, Matrix3x3 *ycbcr_srgb) {
    // YCbCr -> RGB version of the input pixel, for highlights compression, ensure definite positiveness
    float3 rgb = max((float3) (dot(ycbcr_srgb->m[0], input),
                               dot(ycbcr_srgb->m[1], input),
                               dot(ycbcr_srgb->m[2], input)), 0);

    // The filtered image is an estimate of the illuminance
    const float illuminance = gfAb.x * input.x + gfAb.y;
    const float reflectance = input.x / illuminance;

    const float highlightsClipping = min(length(sqrt(2 * rgb)), 1.0);
    const float adjusted_shadows = mix(shadows, 1, highlightsClipping);
    const float gamma = mix(adjusted_shadows, highlights, smoothstep(0.125, 0.75, illuminance));

    // LTM curve computed in Log space
    return pow(illuminance, gamma) * pow(reflectance, detail) / input.x;
}

typedef struct LTMParameters {
    float eps;
    float shadows;
    float highlights;
    float detail[3];
} LTMParameters;

kernel void localToneMappingMaskImage(read_only image2d_t inputImage,
                                      read_only image2d_t lfAbImage,
                                      read_only image2d_t mfAbImage,
                                      read_only image2d_t hfAbImage,
                                      write_only image2d_t ltmMaskImage,
                                      LTMParameters ltmParameters,
                                      Matrix3x3 ycbcr_srgb,
                                      float2 nlf,
                                      sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(ltmMaskImage));
    const float2 pos = convert_float2(imageCoordinates) * inputNorm;

    float3 input = read_imagef(inputImage, imageCoordinates).xyz;
    float2 lfAbSample = read_imagef(lfAbImage, linear_sampler, pos + 0.5f * inputNorm).xy;

    float ltmMultiplier = computeLtmMultiplier(input, lfAbSample, ltmParameters.eps,
                                               ltmParameters.shadows, ltmParameters.highlights,
                                               ltmParameters.detail[0], &ycbcr_srgb);

    if (ltmParameters.detail[1] != 1) {
        float2 mfAbSample = read_imagef(mfAbImage, linear_sampler, pos + 0.5f * inputNorm).xy;
        ltmMultiplier *= computeLtmMultiplier(input, mfAbSample, ltmParameters.eps, 1, 1, ltmParameters.detail[1], &ycbcr_srgb);
    }

    if (ltmParameters.detail[2] != 1) {
        float detail = ltmParameters.detail[2];

        if (detail > 1.0) {
            float dx = (read_imagef(inputImage, imageCoordinates + (int2)(1, 0)).x -
                        read_imagef(inputImage, imageCoordinates - (int2)(1, 0)).x) / 2;
            float dy = (read_imagef(inputImage, imageCoordinates + (int2)(0, 1)).x -
                        read_imagef(inputImage, imageCoordinates - (int2)(0, 1)).x) / 2;

            float noiseThreshold = sqrt(nlf.x + nlf.y * input.x);
            detail = 1 + (detail - 1) * smoothstep(0.5 * noiseThreshold, 2 * noiseThreshold, length((float2) (dx, dy)));
        }

        float2 hfAbSample = read_imagef(hfAbImage, linear_sampler, pos + 0.5f * inputNorm).xy;
        ltmMultiplier *= computeLtmMultiplier(input, hfAbSample, ltmParameters.eps, 1, 1, detail, &ycbcr_srgb);
    }

    write_imagef(ltmMaskImage, imageCoordinates, (float4) (ltmMultiplier, 0, 0, 0));
}

// Denoise filters

float4 denoiseRawRGBAGuidedCov(float4 eps, image2d_t inputImage, int2 imageCoordinates) {
    const float4 input = read_imagef(inputImage, imageCoordinates);

    const int radius = 2;
    const float norm = 1.0 / ((2 * radius + 1) * (2 * radius + 1));

    float3 mean_I = 0;
    float mean_I_rr = 0;
    float mean_I_rg = 0;
    float mean_I_rb = 0;
    float mean_I_gg = 0;
    float mean_I_gb = 0;
    float mean_I_bb = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float4 sampleRGBA = read_imagef(inputImage, imageCoordinates + (int2)(x, y));
            float3 sample = (float3) (sampleRGBA.x, (sampleRGBA.y + sampleRGBA.w) / 2, sampleRGBA.z);
            mean_I += sample;
            mean_I_rr += sample.x * sample.x;
            mean_I_rg += sample.x * sample.y;
            mean_I_rb += sample.x * sample.z;
            mean_I_gg += sample.y * sample.y;
            mean_I_gb += sample.y * sample.z;
            mean_I_bb += sample.z * sample.z;
        }
    }
    mean_I *= norm;
    mean_I_rr *= norm;
    mean_I_rg *= norm;
    mean_I_rb *= norm;
    mean_I_gg *= norm;
    mean_I_gb *= norm;
    mean_I_bb *= norm;

    float var_I_rr = mean_I_rr - mean_I.x * mean_I.x;
    float var_I_rg = mean_I_rg - mean_I.x * mean_I.y;
    float var_I_rb = mean_I_rb - mean_I.x * mean_I.z;
    float var_I_gg = mean_I_gg - mean_I.y * mean_I.y;
    float var_I_gb = mean_I_gb - mean_I.y * mean_I.z;
    float var_I_bb = mean_I_bb - mean_I.z * mean_I.z;

    float var_I_rr_eps = var_I_rr + 0.2 * eps.x * input.x;
    float var_I_gg_eps = var_I_gg + 0.2 * eps.y * (input.y + input.w) / 2;
    float var_I_bb_eps = var_I_bb + 0.2 * eps.z * input.z;

    float invrr = var_I_gg_eps * var_I_bb_eps - var_I_gb     * var_I_gb;
    float invrg = var_I_gb     * var_I_rb     - var_I_rg     * var_I_bb_eps;
    float invrb = var_I_rg     * var_I_gb     - var_I_gg_eps * var_I_rb;
    float invgg = var_I_rr_eps * var_I_bb_eps - var_I_rb     * var_I_rb;
    float invgb = var_I_rb     * var_I_rg     - var_I_rr_eps * var_I_gb;
    float invbb = var_I_rr_eps * var_I_gg_eps - var_I_rg     * var_I_rg;

    float invCovDet = 1 / (invrr * var_I_rr_eps + invrg * var_I_rg + invrb * var_I_rb);

    invrr *= invCovDet;
    invrg *= invCovDet;
    invrb *= invCovDet;
    invgg *= invCovDet;
    invgb *= invCovDet;
    invbb *= invCovDet;

    // Compute the result

    // covariance of (I, p) in each local patch.
    float3 cov_Ip_r = (float3) (var_I_rr, var_I_rg, var_I_rb);
    float3 cov_Ip_g = (float3) (var_I_rg, var_I_gg, var_I_gb);
    float3 cov_Ip_b = (float3) (var_I_rb, var_I_gb, var_I_bb);

    float3 a_r = invrr * cov_Ip_r + invrg * cov_Ip_g + invrb * cov_Ip_b;
    float3 a_g = invrg * cov_Ip_r + invgg * cov_Ip_g + invgb * cov_Ip_b;
    float3 a_b = invrb * cov_Ip_r + invgb * cov_Ip_g + invbb * cov_Ip_b;

    float3 b = mean_I - a_r * mean_I.x - a_g * mean_I.y - a_b * mean_I.z; // Eqn. (15) in the paper;

    return (float4) (a_r * input.x + a_g * input.y + a_b * input.z + b,
                     a_r.y * input.x + a_g.y * input.w + a_b.y * input.z + b.y);
}

constant float sobelX[3][3] = {
    { -1, -2, -1 },
    {  0,  0,  0 },
    {  1,  2,  1 },
};
constant float sobelY[3][3] = {
    { -1,  0,  1 },
    { -2,  0,  2 },
    { -1,  0,  1 },
};

kernel void sobelFilterImage(read_only image2d_t inputImage, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float3 valueX = 0;
    float3 valueY = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            float3 sample = read_imagef(inputImage, imageCoordinates + (int2)(x - 1, y - 1)).xyz;
            valueX += sobelX[y][x] * sample;
            valueY += sobelY[y][x] * sample;
        }
    }

    write_imagef(outputImage, imageCoordinates, (float4) (sqrt(valueX * valueX + valueY * valueY), 0));
}

kernel void desaturateEdges(read_only image2d_t inputImage, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float d2x = read_imagef(inputImage, imageCoordinates + (int2)(-1, 0)).x -
                2 * read_imagef(inputImage, imageCoordinates + (int2)(0, 0)).x +
                read_imagef(inputImage, imageCoordinates + (int2)( 1, 0)).x;

    float d2y = read_imagef(inputImage, imageCoordinates + (int2)(0, -1)).x -
                2 * read_imagef(inputImage, imageCoordinates + (int2)(0, 0)).x +
                read_imagef(inputImage, imageCoordinates + (int2)(0,  1)).x;

    float3 pixel = read_imagef(inputImage, imageCoordinates).xyz;
    float desaturate = 1 - smoothstep(0.25, 0.5, 50 * (d2x * d2x + d2y * d2y));

    write_imagef(outputImage, imageCoordinates, (float4) (pixel.x, desaturate * pixel.yz, 0));
}

constant float laplacian[3][3] = {
    {  1, -2,  1 },
    { -2,  4, -2 },
    {  1, -2,  1 },
};

kernel void laplacianFilterImage(read_only image2d_t inputImage, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float3 value = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            float3 sample = read_imagef(inputImage, imageCoordinates + (int2)(x - 1, y - 1)).xyz;
            value += laplacian[y][x] * sample;
        }
    }

    write_imagef(outputImage, imageCoordinates, (float4) (value, 0));
}

kernel void noiseStatistics(read_only image2d_t inputImage, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    int radius = 2;
    int count = (2 * radius + 1) * (2 * radius + 1);

    float3 sum = 0;
    float3 sumSq = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float3 inputSample = read_imagef(inputImage, imageCoordinates + (int2)(x, y)).xyz;
            sum += inputSample;
            sumSq += inputSample * inputSample;
        }
    }
    float3 mean = sum / count;
    float3 var = (sumSq - (sum * sum) / count) / count;

    write_imagef(outputImage, imageCoordinates, (float4) (mean.x, var));
}

float4 readRAWQuad(read_only image2d_t rawImage, int2 imageCoordinates, constant const int2 *offsets) {
    const int2 r = offsets[raw_red];
    const int2 g = offsets[raw_green];
    const int2 b = offsets[raw_blue];
    const int2 g2 = offsets[raw_green2];

    float red    = read_imagef(rawImage, 2 * imageCoordinates + r).x;
    float green  = read_imagef(rawImage, 2 * imageCoordinates + g).x;
    float blue   = read_imagef(rawImage, 2 * imageCoordinates + b).x;
    float green2 = read_imagef(rawImage, 2 * imageCoordinates + g2).x;

    return (float4) (red, green, blue, green2);
}

kernel void rawNoiseStatistics(read_only image2d_t inputImage, int bayerPattern, write_only image2d_t meanImage, write_only image2d_t varImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    constant const int2* offsets = bayerOffsets[bayerPattern];

    int radius = 2;
    int count = (2 * radius + 1) * (2 * radius + 1);

    float4 sum = 0;
    float4 sumSq = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float4 inputSample = readRAWQuad(inputImage, imageCoordinates + (int2)(x, y), offsets);
            sum += inputSample;
            sumSq += inputSample * inputSample;
        }
    }
    float4 mean = sum / count;
    float4 var = (sumSq - (sum * sum) / count) / count;

    write_imagef(meanImage, imageCoordinates, mean);
    write_imagef(varImage, imageCoordinates, var);
}

/// ---- Image Sharpening ----

float3 gaussianBlur(float radius, image2d_t inputImage, int2 imageCoordinates) {
    const int kernelSize = (int) (2 * ceil(2.5 * radius) + 1);

    float3 blurred_pixel = 0;
    float3 kernel_norm = 0;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++) {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++) {
            float kernelWeight = native_exp(-((float)(x * x + y * y) / (2 * radius * radius)));
            blurred_pixel += kernelWeight * read_imagef(inputImage, imageCoordinates + (int2)(x, y)).xyz;
            kernel_norm += kernelWeight;
        }
    }
    return blurred_pixel / kernel_norm;
}

kernel void gaussianBlurImage(read_only image2d_t inputImage, float radius, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float3 value = gaussianBlur(radius, inputImage, imageCoordinates);

    write_imagef(outputImage, imageCoordinates, (float4) (value, 0));
}

float gaussianBlurLuma(float radius, image2d_t inputImage, int2 imageCoordinates) {
    const int kernelSize = (int) (2 * ceil(2.5 * radius) + 1);

    float blurred_pixel = 0;
    float kernel_norm = 0;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++) {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++) {
            float kernelWeight = native_exp(-((float)(x * x + y * y) / (2 * radius * radius)));
            blurred_pixel += kernelWeight * read_imagef(inputImage, imageCoordinates + (int2)(x, y)).x;
            kernel_norm += kernelWeight;
        }
    }
    return blurred_pixel / kernel_norm;
}

kernel void highPassLumaImage(read_only image2d_t inputImage, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float3 result = read_imagef(inputImage, imageCoordinates).x - gaussianBlurLuma(2, inputImage, imageCoordinates);
    write_imagef(outputImage, imageCoordinates, (float4) (result, 0));
}

float blueNoiseGenerator(read_only image2d_t blueNoiseImage, int2 imageCoordinates, sampler_t linear_sampler) {
    int2 blueNoiseDimensions = get_image_dim(blueNoiseImage);
    int2 blueNoiseCoordinates = imageCoordinates;

    // Read the noise from the texture
    float2 pos = (convert_float2(blueNoiseCoordinates) + 0.5) / convert_float2(blueNoiseDimensions);
    float blueNoise = read_imagef(blueNoiseImage, linear_sampler, pos).x;

    // Go from a uniform distribution on [0,1] to a symmetric triangular distribution on [-1,1] with maximal density at 0
    blueNoise = mad(blueNoise, 2.0f, -1.0f);
    return sign(blueNoise) * (1.0f - sqrt(1.0f - abs(blueNoise)));
}


kernel void blueNoiseImage(read_only image2d_t inputImage,
                           read_only image2d_t blueNoiseImage,
                           float2 lumaVariance,
                           write_only image2d_t outputImage,
                           sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float blueNoise = blueNoiseGenerator(blueNoiseImage, imageCoordinates, linear_sampler);

    float3 pixel = read_imagef(inputImage, imageCoordinates).xyz;

    // Compute the sigma of the noise from Noise Level Function
    float luma_sigma = sqrt(lumaVariance.x + lumaVariance.y * pixel.x);

    float saturation = length(pixel.yz);
    float darkening = 1 - 0.1 * smoothstep(0.1, 0.2, saturation);

    float3 result = (float3) (darkening * (pixel.x + luma_sigma * blueNoise), pixel.yz);

    write_imagef(outputImage, imageCoordinates, (float4) (result, 0));
}

kernel void sampledConvolutionImage(read_only image2d_t inputImage, int samples, constant const float *weights, write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));

    const float2 inputPos = convert_float2(imageCoordinates) * inputNorm;
    float3 sum = 0;
    float norm = 0;
    for (int i = 0; i < samples; i++) {
        float w = weights[3 * i + 0];
        sum += w * read_imagef(inputImage, linear_sampler, inputPos + ((float2) (weights[3 * i + 1], weights[3 * i + 2]) + 0.5) * inputNorm).xyz;
        norm += w;
    }
    write_imagef(outputImage, imageCoordinates, (float4) (sum / norm, 0));
}

float4 sampledConvolution(read_only image2d_t inputImage, int2 imageCoordinates,
                          float2 inputNorm, sampler_t linear_sampler,
                          int samples, constant float *weights) {
    const float2 inputPos = convert_float2(imageCoordinates) * inputNorm;

    float4 sum = 0;
    float norm = 0;
    for (int i = 0; i < samples; i++) {
        float w = weights[3 * i + 0];
        sum += w * read_imagef(inputImage, linear_sampler, inputPos + ((float2) (weights[3 * i + 1], weights[3 * i + 2]) + 0.5) * inputNorm);
        norm += w;
    }
    return sum / norm;
}

kernel void sampledConvolutionSobel(read_only image2d_t rawImage,
                                    read_only image2d_t sobelImage,
                                    int samples1, constant float *weights1,
                                    int samples2, constant float *weights2,
                                    float2 rawVariance,
                                    write_only image2d_t outputImage,
                                    sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));
    float4 result = sampledConvolution(sobelImage, imageCoordinates, inputNorm, linear_sampler, samples1, weights1);

    float sigma = sqrt(rawVariance.x + rawVariance.y * read_imagef(rawImage, imageCoordinates).x);
    if (length(result.xy) < 4 * sigma) {
        result = sampledConvolution(sobelImage, imageCoordinates, inputNorm, linear_sampler, samples2, weights2);
    }

    write_imagef(outputImage, imageCoordinates, (float4)(copysign(result.zw, result.xy), 0, 0));
}

float3 sharpen(float3 pixel_value, float amount, float radius, image2d_t inputImage, int2 imageCoordinates) {
    float3 dx = read_imagef(inputImage, imageCoordinates + (int2)(1, 0)).xyz - pixel_value;
    float3 dy = read_imagef(inputImage, imageCoordinates + (int2)(0, 1)).xyz - pixel_value;

    // Smart sharpening
    float3 sharpening = amount * smoothstep(0.0, 0.03, length(dx) + length(dy))     // Gradient magnitude thresholding
                               * (1.0 - smoothstep(0.95, 1.0, pixel_value))         // Highlight ringing protection
                               * (0.6 + 0.4 * smoothstep(0.0, 0.1, pixel_value));   // Shadows ringing protection

    float3 blurred_pixel = gaussianBlur(radius, inputImage, imageCoordinates);

    return mix(blurred_pixel, pixel_value, fmax(sharpening, 1.0));
}

/// ---- Tone Curve ----

float3 algebraic(float3 x) {
    return x / sqrt(1 + x * x);
}

float3 __attribute__((overloadable)) sigmoid(float3 x, float s) {
    return 0.5 * (tanh(s * x - 0.3 * s) + 1);
}

float __attribute__((overloadable)) sigmoid(float x, float s) {
    return 0.5 * (tanh(s * x - 0.3 * s) + 1);
}

// This tone curve is designed to mostly match the default curve from DNG files
// TODO: it would be nice to have separate control on highlights and shhadows contrast

float3 __attribute__((overloadable)) toneCurve(float3 x, float s) {
    return (sigmoid(native_powr(0.95 * x, 0.5), s) - sigmoid(0.0, s)) / (sigmoid(1.0, s) - sigmoid(0.0, s));
}

float __attribute__((overloadable)) toneCurve(float x, float s) {
    return (sigmoid(native_powr(0.95 * x, 0.5), s) - sigmoid(0.0, s)) / (sigmoid(1.0, s) - sigmoid(0.0, s));
}

float3 saturationBoost(float3 value, float saturation) {
    // Saturation boost with highlight protection
    const float luma = 0.2126 * value.x + 0.7152 * value.y + 0.0722 * value.z; // BT.709-2 (sRGB) luma primaries
    const float3 clipping = smoothstep(0.75, 2.0, value);
    return mix(luma, value, mix(saturation, 1.0, clipping));
}

float3 desaturateBlacks(float3 value) {
    // Saturation boost with highlight protection
    const float luma = 0.2126 * value.x + 0.7152 * value.y + 0.0722 * value.z; // BT.709-2 (sRGB) luma primaries
    const float desaturate = smoothstep(0.005, 0.04, luma);
    return mix(luma, value, desaturate);
}

float3 contrastBoost(float3 value, float contrast) {
    const float gray = 0.2;
    const float3 clipping = smoothstep(0.9, 2.0, value);
    return mix(gray, value, mix(contrast, 1.0, clipping));
}

// Make sure this struct is in sync with the declaration in demosaic.hpp
typedef struct RGBConversionParameters {
    float contrast;
    float saturation;
    float toneCurveSlope;
    float exposureBias;
    float blacks;
    int localToneMapping;
} RGBConversionParameters;

kernel void convertTosRGB(read_only image2d_t linearImage, read_only image2d_t ltmMaskImage, write_only image2d_t rgbImage,
                          Matrix3x3 transform, RGBConversionParameters parameters) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float3 pixel_value = read_imagef(linearImage, imageCoordinates).xyz;

    // Exposure Bias
    pixel_value *= parameters.exposureBias != 0 ? powr(2.0, parameters.exposureBias) : 1;

    // Saturation
    pixel_value = parameters.saturation != 1.0 ? saturationBoost(pixel_value, parameters.saturation) : pixel_value;

    // Contrast
    pixel_value = parameters.contrast != 1.0 ? contrastBoost(pixel_value, parameters.contrast) : pixel_value;

    // Conversion to target color space, ensure definite positiveness
    float3 rgb = max((float3) (dot(transform.m[0], pixel_value),
                               dot(transform.m[1], pixel_value),
                               dot(transform.m[2], pixel_value)), 0);

    // Local Tone Mapping
    if (parameters.localToneMapping) {
        float ltmBoost = read_imagef(ltmMaskImage, imageCoordinates).x;

        if (ltmBoost > 1) {
            // Modified Naik and Murthy’s method for preserving hue/saturation under luminance changes
            const float luma = 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z; // BT.709-2 (sRGB) luma primaries
            rgb = mix(rgb * ltmBoost, luma < 1 ? 1 - (1.0 - rgb) * (1 - ltmBoost * luma) / (1 - luma) : rgb, min(2 * pow(luma, 0.5), 1));
        } else if (ltmBoost < 1) {
            rgb *= ltmBoost;
        }
    }

    // Tone Curve
    rgb = toneCurve(max(rgb, 0), parameters.toneCurveSlope);

    // Black Level Adjustment
    if (parameters.blacks > 0) {
        rgb = (rgb - parameters.blacks) / (1 - parameters.blacks);
    }

    write_imagef(rgbImage, imageCoordinates, (float4) (clamp(rgb, 0.0, 1.0), 0.0));
}

kernel void convertToGrayscale(read_only image2d_t linearImage, write_only image2d_t grayscaleImage, float3 transform) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float3 pixel_value = read_imagef(linearImage, imageCoordinates).xyz;

    // Conversion to grayscale, ensure definite positiveness
    // float grayscale = sqrt(max(dot(transform, pixel_value), 0));

    float grayscale = toneCurve(max(dot(transform, pixel_value), 0.0), 3.5);

    write_imagef(grayscaleImage, imageCoordinates, (float4) (grayscale, 0, 0, 0));
}

kernel void resample(read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));

    float3 outputPixel = read_imagef(inputImage, linear_sampler, convert_float2(imageCoordinates) * inputNorm + 0.5 * inputNorm).xyz;
    write_imagef(outputImage, imageCoordinates, (float4) (outputPixel, 0.0));
}

// Catmull-Rom interpolation passes by the control points, no prefiltering is needed.
// Despite some of the coefficients being negative, this implementation uses only 9
// linear samples instead of 16 nearest neighbor samples.
void CatmullRomInterpolation(read_only image2d_t src, write_only image2d_t dst, sampler_t linear_sampler) {
    int2 dst_coord = {get_global_id(0), get_global_id(1)};

    float2 dst_size = convert_float2(get_image_dim(dst));
    float2 src_size = convert_float2(get_image_dim(src));

    float2 uv = (convert_float2(dst_coord) + 0.5f) / dst_size;

    float2 src_pos = uv * src_size - 0.5f;

    float2 src_coord;
    float2 frac = fract(src_pos, &src_coord);

    // Compute the Catmull-Rom weights.
    float2 w0 = frac * (-0.5f + frac * (1.0f - 0.5f * frac));
    float2 w1 = 1.0f + frac * frac * (-2.5f + 1.5f * frac);
    float2 w2 = frac * (0.5f + frac * (2.0f - 1.5f * frac));
    float2 w3 = frac * frac * (-0.5f + 0.5f * frac);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / w12;

    // Compute the final UV coordinates we'll use for sampling the texture
    float2 texPos0 = native_divide(src_coord - 0.5f, src_size);
    float2 texPos3 = native_divide(src_coord + 2.5f, src_size);
    float2 texPos12 = native_divide(src_coord + 0.5f + offset12, src_size);

    float4 result = read_imagef(src, linear_sampler, (float2) (texPos0.x, texPos0.y)) * w0.x * w0.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos3.x, texPos0.y)) * w3.x * w0.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos0.x, texPos12.y)) * w0.x * w12.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos3.x, texPos12.y)) * w3.x * w12.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos0.x, texPos3.y)) * w0.x * w3.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += read_imagef(src, linear_sampler, (float2) (texPos3.x, texPos3.y)) * w3.x * w3.y;

    write_imagef(dst, dst_coord, clamp(result, 0.0f, 1.0f));
}

kernel void rescaleImage(read_only image2d_t inputImage,
                         write_only image2d_t outputImage,
                         sampler_t linear_sampler) {
    CatmullRomInterpolation(inputImage, outputImage, linear_sampler);
}
