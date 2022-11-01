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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//#define half float
//#define half2 float2
//#define half3 float3
//#define half4 float4
//#define read_imageh read_imagef
//#define write_imageh write_imagef

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

#define min(a, b) ({__typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;})

#define max(a, b) ({__typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;})

#define abs(a) ({__typeof__(a) _a = (a); \
    _a > 0 ? _a : -_a;})

#ifdef __APPLE__
inline half3 __attribute__((overloadable)) myconvert_half3(float3 val) {
    return (half3) (val.x, val.y, val.z);
}

inline half4 __attribute__((overloadable)) myconvert_half4(float4 val) {
    return (half4) (val.x, val.y, val.z, val.w);
}
#else
#define myconvert_half3(val)    convert_half3(val)

#define myconvert_half4(val)    convert_half4(val)
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
                     max(scaleMul[c] * (read_imagef(rawImage, imageCoordinates + (int2) (o.x, o.y)).x - blackLevel), 0.0));
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

float2 gaussFilteredAbsSobel3x3(read_only image2d_t inputImage, int x, int y) {
    // Average Sobel Filter on a 3x3 raw patch
    float2 sum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            sum += gaussianBlur3x3[j + 1][i + 1] * abs(sobel(inputImage, x + i, y + j));
        }
    }
    return sum;
}

float2 channelCorrelation(read_only image2d_t rawImage, int x, int y) {
    // Estimate the correlation between the green and the color channel on a 3x3 quad patch
    float s_c = 0;
    float2 s_g = 0;
    float s_cc = 0;
    float2 s_gg = 0;
    float2 s_gc = 0;
    float N = 9;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 pos = { x + 2 * i, y + 2 * j };
            float c = read_imagef(rawImage, pos).x;
            float2 g = { (read_imagef(rawImage, pos + (int2)(1, 0)).x + read_imagef(rawImage, pos + (int2)(-1, 0)).x) / 2,
                         (read_imagef(rawImage, pos + (int2)(0, 1)).x + read_imagef(rawImage, pos + (int2)(0, -1)).x) / 2 };

            s_c += c;
            s_cc += c * c;
            s_g += g;
            s_gg += g * g;
            s_gc += c * g;
        }
    }
    float2 cov_cg = (N * s_gc - s_c * s_g);
    float2 var_g = (N * s_gg - s_g * s_g);
    float2 var_c = (N * s_cc - s_c * s_c);

    return var_g > 0 && var_c > 0 ? abs(cov_cg / sqrt(var_g * var_c)) : 1;
}

static constant const float kHighNoiseVariance = 1e-03;

kernel void interpolateGreen(read_only image2d_t rawImage, write_only image2d_t greenImage, int bayerPattern, float2 greenVariance) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const int x = imageCoordinates.x;
    const int y = imageCoordinates.y;

    const int2 g = bayerOffsets[bayerPattern][raw_green];
    int x0 = (y & 1) == (g.y & 1) ? g.x + 1 : g.x;

    if ((x0 & 1) == (x & 1)) {
        float g_left  = read_imagef(rawImage, (int2)(x - 1, y)).x;
        float g_right = read_imagef(rawImage, (int2)(x + 1, y)).x;
        float g_up    = read_imagef(rawImage, (int2)(x, y - 1)).x;
        float g_down  = read_imagef(rawImage, (int2)(x, y + 1)).x;

        float c_xy    = read_imagef(rawImage, (int2)(x, y)).x;

        float c_left  = read_imagef(rawImage, (int2)(x - 2, y)).x;
        float c_right = read_imagef(rawImage, (int2)(x + 2, y)).x;
        float c_up    = read_imagef(rawImage, (int2)(x, y - 2)).x;
        float c_down  = read_imagef(rawImage, (int2)(x, y + 2)).x;

        float c2_top_left = read_imagef(rawImage, (int2)(x - 1, y - 1)).x;
        float c2_top_right = read_imagef(rawImage, (int2)(x + 1, y - 1)).x;
        float c2_bottom_left = read_imagef(rawImage, (int2)(x - 1, y + 1)).x;
        float c2_bottom_right = read_imagef(rawImage, (int2)(x + 1, y + 1)).x;
        float c2_ave = (c2_top_left + c2_top_right + c2_bottom_left + c2_bottom_right) / 4;

        // Estimate gradient intensity and direction
        float g_ave = (g_left + g_right + g_up + g_down) / 4;
        float rawStdDev = sqrt(greenVariance.x + greenVariance.y * g_ave);
        float2 gradient = gaussFilteredAbsSobel3x3(rawImage, x, y);

        // Hamilton-Adams second order Laplacian Interpolation
        float2 g_lf = { (g_left + g_right) / 2, (g_up + g_down) / 2 };
        float2 g_hf = { (2 * c_xy - (c_left + c_right)) / 4, (2 * c_xy - (c_up + c_down)) / 4 };

        // Limit the range of HF correction to something reasonable
        g_hf = clamp(g_hf, - 2 * g_lf, 2 * g_lf);

        // Estimate the pixel's "whiteness"
        float whiteness = clamp(min(c_xy, min(g_ave, c2_ave)) / max(c_xy, max(g_ave, c2_ave)), 0.0, 1.0);

        // Minimum gradient threshold wrt the noise model
        float gradient_threshold = smoothstep(rawStdDev, 4 * rawStdDev, length(gradient));

        // Edges that are in strong highlights tend to grossly overestimate the gradient
        float highlights_edge = 1 - smoothstep(0.25, 1.0, max(c_xy, max(max(c_left, c_right), max(c_up, c_down))));

        // Gradient direction in [0..1]
        float direction = 2 * atan2pi(gradient.y, gradient.x);

        if (greenVariance.y < kHighNoiseVariance) {
            // Bias result towards vertical and horizontal lines
            direction = direction < 0.5 ? mix(direction, 0, 1 - smoothstep(0.3, 0.45, direction))
                                        : mix(direction, 1, smoothstep((1 - 0.45), (1 - 0.3), direction));
        }

        // If the gradient is below threshold interpolate against the grain
        direction = mix(1 - direction, direction, gradient_threshold);

        // Reduce hf_gain when direction is diagonal
        float diagonality = 1 - 0.5 * sin(M_PI_F * direction);

        // Modulate the HF component of the reconstructed green using the whteness and the gradient magnitude
        float hf_gain = diagonality * highlights_edge * gradient_threshold * min(0.5 * whiteness + smoothstep(0.0, 0.3, length(gradient) / M_SQRT2_F), 1.0);
        float2 g_est = g_lf + hf_gain * g_hf;

        // Green pixel estimation
        float sample = mix(g_est.y, g_est.x, direction);

        write_imagef(greenImage, imageCoordinates, sample);
    } else {
        write_imagef(greenImage, imageCoordinates, read_imagef(rawImage, imageCoordinates).x);
    }
}

kernel void interpolateRedBlue(read_only image2d_t rawImage, read_only image2d_t greenImage,
                               write_only image2d_t rgbImage, int bayerPattern,
                               float2 redVariance, float2 blueVariance) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const int x = imageCoordinates.x;
    const int y = imageCoordinates.y;

    const int2 r = bayerOffsets[bayerPattern][raw_red];
    const int2 g = bayerOffsets[bayerPattern][raw_green];
    const int2 b = bayerOffsets[bayerPattern][raw_blue];

    int color = (r.x & 1) == (x & 1) && (r.y & 1) == (y & 1) ? raw_red :
                (g.x & 1) == (x & 1) && (g.y & 1) == (y & 1) ? raw_green :
                (b.x & 1) == (x & 1) && (b.y & 1) == (y & 1) ? raw_blue : raw_green2;

    float green = read_imagef(greenImage, imageCoordinates).x;
    float red;
    float blue;
    switch (color) {
        case raw_red:
        case raw_blue:
        {
            float c1 = read_imagef(rawImage, imageCoordinates).x;

            float g_top_left      = read_imagef(greenImage, (int2)(x - 1, y - 1)).x;
            float g_top_right     = read_imagef(greenImage, (int2)(x + 1, y - 1)).x;
            float g_bottom_left   = read_imagef(greenImage, (int2)(x - 1, y + 1)).x;
            float g_bottom_right  = read_imagef(greenImage, (int2)(x + 1, y + 1)).x;

            float c2_top_left     = read_imagef(rawImage, (int2)(x - 1, y - 1)).x;
            float c2_top_right    = read_imagef(rawImage, (int2)(x + 1, y - 1)).x;
            float c2_bottom_left  = read_imagef(rawImage, (int2)(x - 1, y + 1)).x;
            float c2_bottom_right = read_imagef(rawImage, (int2)(x + 1, y + 1)).x;
            float c2_ave = (c2_top_left + c2_top_right + c2_bottom_left + c2_bottom_right) / 4;

            float gc_top_left     = g_top_left     - c2_top_left;
            float gc_top_right    = g_top_right    - c2_top_right;
            float gc_bottom_left  = g_bottom_left  - c2_bottom_left;
            float gc_bottom_right = g_bottom_right - c2_bottom_right;

            float2 variance = color == raw_red ? redVariance : blueVariance;

            // Estimate the gradient direction taking into account the raw noise model
            float rawStdDev = sqrt(variance.x + variance.y * c2_ave);
            float2 dv = (float2) (fabs(c2_top_left - c2_bottom_right), fabs(c2_top_right - c2_bottom_left));
            float direction = 2 * atan2pi(dv.y, dv.x);
            float gradient_threshold = smoothstep(rawStdDev, 4 * rawStdDev, length(dv));
            // If the gradient is below threshold interpolate against the grain
            float alpha = mix(1 - direction, direction, gradient_threshold);
            float c2 = green - mix((gc_top_right + gc_bottom_left) / 2,
                                   (gc_top_left + gc_bottom_right) / 2, alpha);

            if (color == raw_red) {
                red = c1;
                blue = c2;
            } else {
                blue = c1;
                red = c2;
            }
        }
        break;

        case raw_green:
        case raw_green2:
        {
            float g_left    = read_imagef(greenImage, (int2)(x - 1, y)).x;
            float g_right   = read_imagef(greenImage, (int2)(x + 1, y)).x;
            float g_up      = read_imagef(greenImage, (int2)(x, y - 1)).x;
            float g_down    = read_imagef(greenImage, (int2)(x, y + 1)).x;

            float c1_left   = g_left  - read_imagef(rawImage, (int2)(x - 1, y)).x;
            float c1_right  = g_right - read_imagef(rawImage, (int2)(x + 1, y)).x;
            float c2_up     = g_up    - read_imagef(rawImage, (int2)(x, y - 1)).x;
            float c2_down   = g_down  - read_imagef(rawImage, (int2)(x, y + 1)).x;

            float c1 = green - (c1_left + c1_right) / 2;
            float c2 = green - (c2_up + c2_down) / 2;

            if (color == (bayerPattern == bggr || bayerPattern == grbg ? raw_green : raw_green2)) {
                red = c1;
                blue = c2;
            } else {
                blue = c1;
                red = c2;
            }
        }
        break;
    }

    write_imagef(rgbImage, imageCoordinates, (float4)(red, green, blue, 0));
}

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

kernel void blendHighlightsImage(read_only image2d_t inputImage, float clip, write_only image2d_t outputImage) {
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

#define readImage(image, pos)  read_imageh(image, pos).xyz;

kernel void medianFilterImage3x3(read_only image2d_t inputImage, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    typedef half3 medianPixelType;

    half3 median = fast_median3x3(inputImage, imageCoordinates);

    write_imageh(denoisedImage, imageCoordinates, (half4) (median, 0));
}

#undef readImage

#define readImage(image, pos)  read_imageh(image, pos);

kernel void medianFilterImage3x3x4(read_only image2d_t inputImage, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    typedef half4 medianPixelType;

    half4 median = fast_median3x3(inputImage, imageCoordinates);

    write_imageh(denoisedImage, imageCoordinates, median);
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
                                           int desaturateShadows, write_only image2d_t denoisedImage) {
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

    half4 sigma = sqrt(myconvert_half4(rawVariance) * sample);
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

    half4 sigma = sqrt(myconvert_half4(rawVariance) * sample);
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

float2 signedGaussFilteredSobel3x3(read_only image2d_t inputImage, int x, int y) {
    // Average Sobel Filter on a 3x3 raw patch
    float2 sum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            sum += gaussianBlur3x3[j + 1][i + 1] * sobel(inputImage, x + i, y + j);
        }
    }
    return sum;
}

half tunnel(half x, half y, half angle, half sigma) {
    half a = x * cos(angle) + y * sin(angle);
    return exp(-(a * a) / sigma);
}

kernel void denoiseImage(read_only image2d_t inputImage, float3 var_a, float3 var_b, float chromaBoost, float gradientBoost, write_only image2d_t denoisedImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    const half3 inputYCC = read_imageh(inputImage, imageCoordinates).xyz;

    half3 sigma = myconvert_half3(sqrt(var_a + var_b * inputYCC.x));

    float2 gradient = signedGaussFilteredSobel3x3(inputImage, imageCoordinates.x, imageCoordinates.y);
    half angle = atan2(gradient.y, gradient.x);
    half magnitude = length(gradient);
    half edge = smoothstep(4, 16, magnitude / sigma.x);
    // TODO: make this a tunable parameter
    half flat = 0; // 1 - smoothstep(1, 4, magnitude / sigma.x);

    const int size = gradientBoost > 1 ? 4 : 2;

    half3 filtered_pixel = 0;
    half3 kernel_norm = 0;
    for (int y = -size; y <= size; y++) {
        for (int x = -size; x <= size; x++) {
            half3 inputSampleYCC = read_imageh(inputImage, imageCoordinates + (int2)(x, y)).xyz;

            half3 inputDiff = (inputSampleYCC - inputYCC) / sigma;

            half w = (half) mix(1, tunnel(x, y, angle, 0.25h), edge);
            half lumaWeight = w * (1 - step(1 + (half) gradientBoost * edge + flat, abs(inputDiff.x)));
            half chromaWeight = abs(x) <= 2 && abs(y) <= 2 ? 1 - step((half) chromaBoost, length(inputDiff)) : 0;
            half3 sampleWeight = (half3) (lumaWeight, chromaWeight, chromaWeight);

            filtered_pixel += sampleWeight * inputSampleYCC;
            kernel_norm += sampleWeight;
        }
    }
    half3 denoisedPixel = filtered_pixel / kernel_norm;

    write_imageh(denoisedImage, imageCoordinates, (half4) (denoisedPixel, 0.0));
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

//#define USE_LANCZOS_INTERPOLATION 1

/*
 Lanczos interpolation retains more high frequency content in the upper pyramid layers
 */

#if USE_LANCZOS_INTERPOLATION
float lanczos2(float2 p) {
    float x = length(p);
    return x == 0 ? 0 : x < 2 ? sin(M_PI_F * x) * sin(M_PI_F * x / 2) / ((M_PI_F * x) * (M_PI_F * x / 2)) : 0;
}

float3 lanczosInterpolation(read_only image2d_t inputImage, const float2 input_pos, const float2 input_norm, sampler_t linear_sampler) {
    float3 outputPixel = 0;
    float norm = 0;
    for (float y = -2; y <= 2; y += 1) {
        for (float x = -2; x <= 2; x += 1) {
            float k = lanczos2((float2)(x, y));
            norm += k;
            outputPixel += k * read_imagef(inputImage, linear_sampler, input_pos + (float2)(x, y) * input_norm).xyz;
        }
    }
    outputPixel /= norm;
    return outputPixel;
}
#endif

kernel void downsampleImage(read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 output_pos = (int2) (get_global_id(0), get_global_id(1));
#if USE_LANCZOS_INTERPOLATION
    float3 outputPixel = 0;
    float norm = 0;
    for (int y = -2; y < 2; y++) {
        for (int x = -2; x < 2; x++) {
            float k = lanczos2((float2)(x, y) - 0.5);
            norm += k;
            outputPixel += k * read_imagef(inputImage, output_pos * 2 + (int2)(x, y)).xyz;
        }
    }
    outputPixel /= norm;
    write_imagef(outputImage, output_pos, (float4) (outputPixel, 0));
#else
    const float2 input_norm = 1.0 / convert_float2(get_image_dim(outputImage));
    const float2 input_pos = (convert_float2(output_pos) + 0.5) * input_norm;

    // Sub-Pixel Sampling Location
    const float2 s = 0.5 * input_norm;
    float3 outputPixel = read_imagef(inputImage, linear_sampler, input_pos + (float2)(-s.x, -s.y)).xyz;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)( s.x, -s.y)).xyz;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)(-s.x,  s.y)).xyz;
    outputPixel +=       read_imagef(inputImage, linear_sampler, input_pos + (float2)( s.x,  s.y)).xyz;
    write_imagef(outputImage, output_pos, (float4) (0.25 * outputPixel, 0));
#endif
}

float3 applyTransform(float3 value, Matrix3x3 *transform) {
    return (float3) (dot(transform->m[0], value), dot(transform->m[1], value), dot(transform->m[2], value));
}

kernel void reassembleImage(read_only image2d_t inputImageDenoised0, read_only image2d_t inputImage1,
                            read_only image2d_t inputImageDenoised1, float sharpening, float2 nlf,
                            write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 output_pos = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));
    const float2 input_pos = (convert_float2(output_pos) + 0.5) * inputNorm;

    float3 inputPixelDenoised0 = read_imagef(inputImageDenoised0, output_pos).xyz;

#if USE_LANCZOS_INTERPOLATION
    float3 inputPixel1 = lanczosInterpolation(inputImage1, input_pos, inputNorm, linear_sampler);
    float3 inputPixelDenoised1 = lanczosInterpolation(inputImageDenoised1, input_pos, inputNorm, linear_sampler);
#else
    float3 inputPixel1 = read_imagef(inputImage1, linear_sampler, input_pos).xyz;
    float3 inputPixelDenoised1 = read_imagef(inputImageDenoised1, linear_sampler, input_pos).xyz;
#endif

    float3 denoisedPixel = inputPixelDenoised0 - (inputPixel1 - inputPixelDenoised1);

    if (sharpening > 1.0) {
        float dx = (read_imagef(inputImageDenoised1, linear_sampler, input_pos + (float2)(1, 0) * inputNorm).x -
                    read_imagef(inputImageDenoised1, linear_sampler, input_pos - (float2)(1, 0) * inputNorm).x) / 2;
        float dy = (read_imagef(inputImageDenoised1, linear_sampler, input_pos + (float2)(0, 1) * inputNorm).x -
                    read_imagef(inputImageDenoised1, linear_sampler, input_pos - (float2)(0, 1) * inputNorm).x) / 2;

        float threshold = 0.25 * sqrt(nlf.x + nlf.y * inputPixelDenoised1.x);
        float detail = smoothstep(0.25 * threshold, threshold, length((float2) (dx, dy)))
                       * (1.0 - smoothstep(0.95, 1.0, denoisedPixel.x))          // Highlights ringing protection
                       * (0.6 + 0.4 * smoothstep(0.0, 0.1, denoisedPixel.x));    // Shadows ringing protection
        sharpening = 1 + (sharpening - 1) * detail;
    }

    // Sharpen all components
    denoisedPixel = mix(inputPixelDenoised1, denoisedPixel, sharpening);

    write_imagef(outputImage, output_pos, (float4) (denoisedPixel, 0));
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
            float4 sampleWeight = gaussianBlur5x5[y + 2][x + 2] * (1 - step(1, diff));
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

    float4 denoisedPixel = denoiseRawRGBAPatch(rawVariance, inputImage, imageCoordinates);

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

kernel void sobelFilterImage(read_only image2d_t inputImage, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

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

kernel void laplacianFilterImage(read_only image2d_t inputImage, write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    constant float laplacian[3][3] = {
        {  1, -2,  1 },
        { -2,  4, -2 },
        {  1, -2,  1 },
    };

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

float4 readRAWQuad(read_only image2d_t rawImage, int2 imageCoordinates, constant const int2 offsets[4]) {
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

float blueNoiseGenerator(read_only image2d_t blueNoiseImage, int2 blueNoiseCoordinates) {
    // Generate a "random" set of coordinates for the blueNoiseImage
    blueNoiseCoordinates ^= (blueNoiseCoordinates << 21);
    blueNoiseCoordinates ^= (blueNoiseCoordinates >> 35);
    blueNoiseCoordinates ^= (blueNoiseCoordinates << 4);
    blueNoiseCoordinates = blueNoiseCoordinates & (get_image_dim(blueNoiseImage) - 1);  // Assumed to be a power of 2

    // Read the noise from the texture
    float blueNoise = read_imagef(blueNoiseImage, blueNoiseCoordinates).x;

    // Go from a uniform distribution on [0,1] to a symmetric triangular distribution on [-1,1] with maximal density at 0
    blueNoise = mad(blueNoise, 2.0f, -1.0f);
    return sign(blueNoise) * (1.0f - sqrt(1.0f - abs(blueNoise)));
}

static constant const float3 rgbToY = { 0.2126, 0.7152, 0.0722 };

kernel void blueNoiseImage(read_only image2d_t inputImage,
                           read_only image2d_t blueNoiseImage,
                           float2 lumaVariance,
                           write_only image2d_t outputImage) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));

    float blueNoise = blueNoiseGenerator(blueNoiseImage, imageCoordinates);

    float3 pixel = read_imagef(inputImage, imageCoordinates).xyz;

    // Compute the sigma of the noise from Noise Level Function
    float luma = dot(pixel, rgbToY);
    float luma_sigma = sqrt(lumaVariance.x + lumaVariance.y * luma);

    float3 result = pixel + 0.5 * luma_sigma * blueNoise;

    write_imagef(outputImage, imageCoordinates, (float4) (result, 0));
}

kernel void sampledConvolution(read_only image2d_t inputImage, int samples, constant float weights[][3], write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));

    const float2 inputPos = convert_float2(imageCoordinates) * inputNorm;
    float3 sum = 0;
    float norm = 0;
    for (int i = 0; i < samples; i++) {
        float w = weights[i][0];
        sum += w * read_imagef(inputImage, linear_sampler, inputPos + ((float2) (weights[i][1], weights[i][2]) + 0.5) * inputNorm).xyz;
        norm += w;
    }
    write_imagef(outputImage, imageCoordinates, (float4) (sum / norm, 0));
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

float3 superSigma(float3 x) {
    float z = 2;
    return copysign(powr(tanh(powr(abs(0.6 * x), z)), 1/z), x);
}

float3 sigmoid(float3 x, float s) {
    return 0.5 * (tanh(s * x - 0.3 * s) + 1);
}

// This tone curve is designed to mostly match the default curve from DNG files
// TODO: it would be nice to have separate control on highlights and shhadows contrast

float3 toneCurve(float3 x, float s) {
    return (sigmoid(native_powr(0.95 * x, 0.5), s) - sigmoid(0, s)) / (sigmoid(1, s) - sigmoid(0, s));
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

kernel void resample(read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 inputNorm = 1.0 / convert_float2(get_image_dim(outputImage));

    float3 outputPixel = read_imagef(inputImage, linear_sampler, convert_float2(imageCoordinates) * inputNorm + 0.5 * inputNorm).xyz;
    write_imagef(outputImage, imageCoordinates, (float4) (outputPixel, 0.0));
}
