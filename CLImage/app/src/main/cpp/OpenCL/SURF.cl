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

// TODO: generate the wavelet parameters dynamically in the shader
typedef struct SurfHF {
    int8 p_dx[2];
    int8 p_dy[2];
    int8 p_dxy[4];
} SurfHF;

float integralRectangle(float topRight, float topLeft, float bottomRight, float bottomLeft) {
    // Use Signed Offset Pixel Representation to improve Integral Image precision, see Integral Image code below
    return 0.5 + (topRight - topLeft) - (bottomRight - bottomLeft);
}

/*
 NOTE: calcHaarPatternDx and calcHaarPatternDy have been hand optimized to avoid loading data from repeated offsets,
       and the redundant offsets themselves have been removed from the SurfHF struct.
 */
float calcHaarPatternDx(read_only image2d_t inputImage, const int2 p, constant int8 *dp, float w) {
    float r02 = read_imagef(inputImage, p + dp[0].lo.lo).x;
    float r07 = read_imagef(inputImage, p + dp[0].lo.hi).x;
    float r32 = read_imagef(inputImage, p + dp[0].hi.lo).x;
    float r37 = read_imagef(inputImage, p + dp[0].hi.hi).x;

    float r62 = read_imagef(inputImage, p + dp[1].lo.lo).x;
    float r67 = read_imagef(inputImage, p + dp[1].lo.hi).x;

    float r92 = read_imagef(inputImage, p + dp[1].hi.lo).x;
    float r97 = read_imagef(inputImage, p + dp[1].hi.hi).x;

    return w * (integralRectangle(r97, r92, r07, r02) - 3 * integralRectangle(r67, r62, r37, r32));
}

float calcHaarPatternDy(read_only image2d_t inputImage, const int2 p, constant int8 *dp, float w) {
    float r20 = read_imagef(inputImage, p + dp[0].lo.lo).x;
    float r23 = read_imagef(inputImage, p + dp[0].lo.hi).x;
    float r26 = read_imagef(inputImage, p + dp[1].lo.lo).x;
    float r29 = read_imagef(inputImage, p + dp[1].hi.lo).x;

    float r70 = read_imagef(inputImage, p + dp[0].hi.lo).x;
    float r73 = read_imagef(inputImage, p + dp[0].hi.hi).x;
    float r76 = read_imagef(inputImage, p + dp[1].lo.hi).x;
    float r79 = read_imagef(inputImage, p + dp[1].hi.hi).x;

    return w * (integralRectangle(r79, r70, r29, r20) - 3 * integralRectangle(r76, r73, r26, r23));
}

float calcHaarPatternDxy(read_only image2d_t inputImage, const int2 p, constant int8 *dp, float w) {
    const float w4[4] = { w, -w, -w, w };
    float d = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int8 v = dp[k];

        float p0 = read_imagef(inputImage, p + v.lo.lo /* p0 */).x;
        float p1 = read_imagef(inputImage, p + v.lo.hi /* p1 */).x;
        float p2 = read_imagef(inputImage, p + v.hi.lo /* p2 */).x;
        float p3 = read_imagef(inputImage, p + v.hi.hi /* p3 */).x;

        d += w4[k] * integralRectangle(p0, p1, p2, p3);
    }
    return d;
}

kernel void calcDetAndTrace(read_only image2d_t sumImage,
                            write_only image2d_t detImage,
                            write_only image2d_t traceImage,
                            int sampleStep,
                            float2 w,
                            int2 margin,
                            constant SurfHF *surfHFData) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const int2 p = imageCoordinates * sampleStep;

    const float dx = calcHaarPatternDx(sumImage, p, surfHFData->p_dx, w.x);
    const float dy = calcHaarPatternDy(sumImage, p, surfHFData->p_dy, w.x);
    const float dxy = calcHaarPatternDxy(sumImage, p, surfHFData->p_dxy, w.y);

    write_imagef(detImage, imageCoordinates + margin, dx * dy - 0.81f * dxy * dxy);
    write_imagef(traceImage, imageCoordinates + margin, dx + dy);
}

float2 detAndTrace(read_only image2d_t sumImage, const int2 p, constant SurfHF *surfHFData, const float2 w) {
    const float dx = calcHaarPatternDx(sumImage, p, surfHFData[0].p_dx, w.x);
    const float dy = calcHaarPatternDy(sumImage, p, surfHFData[0].p_dy, w.x);
    const float dxy = calcHaarPatternDxy(sumImage, p, surfHFData[0].p_dxy, w.y);
    return (float2) (dx * dy - 0.81f * dxy * dxy, dx + dy);
}

kernel void calcDetAndTrace4(read_only image2d_t sumImage,
                             write_only image2d_t detImage0, write_only image2d_t detImage1, write_only image2d_t detImage2, write_only image2d_t detImage3,
                             write_only image2d_t traceImage0, write_only image2d_t traceImage1, write_only image2d_t traceImage2, write_only image2d_t traceImage3,
                             int sampleStep,
                             float8 w,
                             int4 margin,
                             constant SurfHF *surfHFData) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const int2 p = imageCoordinates * sampleStep;

    float2 detAndTrace0 = detAndTrace(sumImage, p, &surfHFData[0], w.lo.lo);
    float2 detAndTrace1 = detAndTrace(sumImage, p, &surfHFData[1], w.lo.hi);
    float2 detAndTrace2 = detAndTrace(sumImage, p, &surfHFData[2], w.hi.lo);
    float2 detAndTrace3 = detAndTrace(sumImage, p, &surfHFData[3], w.hi.hi);

    write_imagef(detImage0, imageCoordinates + margin.x, (float4)(detAndTrace0.x, 0, 0, 0));
    write_imagef(traceImage0, imageCoordinates + margin.x, (float4)(detAndTrace0.y, 0, 0, 0));

    write_imagef(detImage1, imageCoordinates + margin.y, (float4)(detAndTrace1.x, 0, 0, 0));
    write_imagef(traceImage1, imageCoordinates + margin.y, (float4)(detAndTrace1.y, 0, 0, 0));

    write_imagef(detImage2, imageCoordinates + margin.z, (float4)(detAndTrace2.x, 0, 0, 0));
    write_imagef(traceImage2, imageCoordinates + margin.z, (float4)(detAndTrace2.y, 0, 0, 0));

    write_imagef(detImage3, imageCoordinates + margin.w, (float4)(detAndTrace3.x, 0, 0, 0));
    write_imagef(traceImage3, imageCoordinates + margin.w, (float4)(detAndTrace3.y, 0, 0, 0));
}

inline float determinant(const float3 A0, const float3 A1, const float3 A2) {
    // The cross product computes the 2x2 sub-determinants
    return dot(A0, cross(A1, A2));
}

// Simple Cramer's rule solver
inline float3 solve3x3(const float3 A[3], const float3 B) {
    float det = determinant(A[0], A[1], A[2]);
    if (det == 0) {
        return 0;
    }
    return (float3) (
        determinant(B,    A[1], A[2]),
        determinant(A[0], B,    A[2]),
        determinant(A[0], A[1], B)
    ) / det;
}

typedef struct KeyPoint {
    struct {
        float x, y;
    } pt;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;
} KeyPoint;

#define N9(_idx, _off_y, _off_x) read_imagef(detImage ## _idx, p + (int2) (_off_x, _off_y)).x

bool interpolateKeypoint(read_only image2d_t detImage0, read_only image2d_t detImage1, read_only image2d_t detImage2,
                         int2 p, int dx, int dy, int ds, KeyPoint* kpt) {
    float3 B = {
        -(N9(1, 0, 1) - N9(1,  0, -1)) / 2, // Negative 1st deriv with respect to x
        -(N9(1, 1, 0) - N9(1, -1,  0)) / 2, // Negative 1st deriv with respect to y
        -(N9(2, 0, 0) - N9(0,  0,  0)) / 2  // Negative 1st deriv with respect to s
    };
    float3 A[3] = {
        {  N9(1,  0, -1) - 2 * N9(1,  0,  0) + N9(1,  0, 1),                           // 2nd deriv x, x
          (N9(1,  1,  1) -     N9(1,  1, -1) - N9(1, -1, 1) + N9(1, -1, -1)) / 4,      // 2nd deriv x, y
          (N9(2,  0,  1) -     N9(2,  0, -1) - N9(0,  0, 1) + N9(0,  0, -1)) / 4 },    // 2nd deriv x, s
        { (N9(1,  1,  1) -     N9(1,  1, -1) - N9(1, -1, 1) + N9(1, -1, -1)) / 4,      // 2nd deriv x, y
           N9(1, -1,  0) - 2 * N9(1,  0,  0) + N9(1,  1, 0),                           // 2nd deriv y, y
          (N9(2,  1,  0) -     N9(2, -1,  0) - N9(0,  1, 0) + N9(0, -1,  0)) / 4 },    // 2nd deriv y, s
        { (N9(2,  0,  1) -     N9(2,  0, -1) - N9(0,  0, 1) + N9(0,  0, -1)) / 4,      // 2nd deriv x, s
          (N9(2,  1,  0) -     N9(2, -1,  0) - N9(0,  1, 0) + N9(0, -1,  0)) / 4,      // 2nd deriv y, s
           N9(0,  0,  0) - 2 * N9(1,  0,  0) + N9(2,  0, 0) }                          // 2nd deriv s, s
    };
    float3 x = solve3x3(A, B);

    if (!all(x == 0) && all(fabs(x) <= 1)) {
        kpt->pt.x += x.x * dx;
        kpt->pt.y += x.y * dy;
        kpt->size = (float) rint(kpt->size + x.z * ds);
        return true;
    }
    return false;
}

#define KeyPointMaxima_MaxCount 64000
typedef struct KeyPointMaxima {
    int count;
    KeyPoint keyPoints[KeyPointMaxima_MaxCount];
} KeyPointMaxima;

kernel void findMaximaInLayer(read_only image2d_t detImage0, read_only image2d_t detImage1, read_only image2d_t detImage2,
                              read_only image2d_t traceImage, int3 sizes, global KeyPointMaxima* keypoints,
                              int margin, int octave, float hessianThreshold, int sampleStep) {
    const int2 p = (int2) (get_global_id(0), get_global_id(1)) + margin;

    const int size = sizes.y;

    const float val0 = N9(1, 0, 0);
    
    if (val0 > hessianThreshold) {
        /* Coordinates for the start of the wavelet in the sum image. There
           is some integer division involved, so don't try to simplify this
           (cancel out sampleStep) without checking the result is the same */
        int sum_y = sampleStep * (p.y - (size / 2) / sampleStep);
        int sum_x = sampleStep * (p.x - (size / 2) / sampleStep);

        /* The 3x3x3 neighbouring samples around the maxima.
           The maxima is included at N9[1][0][0] */

        /* Non-maxima suppression. val0 is at N9[1][0][0] */
        if (val0 > N9(0, -1, -1) && val0 > N9(0, -1, 0) && val0 > N9(0, -1, 1) &&
            val0 > N9(0,  0, -1) && val0 > N9(0,  0, 0) && val0 > N9(0,  0, 1) &&
            val0 > N9(0,  1, -1) && val0 > N9(0,  1, 0) && val0 > N9(0,  1, 1) &&
            val0 > N9(1, -1, -1) && val0 > N9(1, -1, 0) && val0 > N9(1, -1, 1) &&
            val0 > N9(1,  0, -1)                        && val0 > N9(1,  0, 1) &&
            val0 > N9(1,  1, -1) && val0 > N9(1,  1, 0) && val0 > N9(1,  1, 1) &&
            val0 > N9(2, -1, -1) && val0 > N9(2, -1, 0) && val0 > N9(2, -1, 1) &&
            val0 > N9(2,  0, -1) && val0 > N9(2,  0, 0) && val0 > N9(2,  0, 1) &&
            val0 > N9(2,  1, -1) && val0 > N9(2,  1, 0) && val0 > N9(2,  1, 1))
        {
            /* Calculate the wavelet center coordinates for the maxima */
            float center_y = sum_y + (size - 1) * 0.5f;
            float center_x = sum_x + (size - 1) * 0.5f;
            KeyPoint kpt = {{center_x, center_y}, (float)sizes.y, -1, val0, octave, read_imagef(traceImage, p).x > 0 };

            /* Interpolate maxima location within the 3x3x3 neighbourhood  */
            int ds = size - sizes.x;
            int interp_ok = interpolateKeypoint(detImage0, detImage1, detImage2, p, sampleStep, sampleStep, ds, &kpt);

            /* Sometimes the interpolation step gives a negative size etc. */
            if (interp_ok) {
                int ind = atomic_inc(&keypoints->count);
                if (ind < KeyPointMaxima_MaxCount) {
                    keypoints->keyPoints[ind] = kpt;
                }
            }
        }
    }
}

// Integral Image

#define LOCAL_SUM_SIZE      8

kernel void integral_sum_cols_image(read_only image2d_t sourceImage, global float *buf_ptr, int buf_width) {
    const int lid = get_local_id(0);
    const int gid = get_group_id(0);
    const int x = get_global_id(0);

    local float lm_sum[LOCAL_SUM_SIZE][LOCAL_SUM_SIZE];

    float accum = 0;
    for (int y = 0; y < get_image_height(sourceImage); y += LOCAL_SUM_SIZE) {
#pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++) {
            // Use Signed Offset Pixel Representation to improve Integral Image precision
            // See: Hensley et al.: "Fast Summed-Area Table Generation and its Applications".
            accum += read_imagef(sourceImage, (int2)(x, y + yin)).x - 0.5;
            lm_sum[yin][lid] = accum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int buf_index = buf_width * LOCAL_SUM_SIZE * gid + (lid + y);
#pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, buf_index += buf_width) {
            buf_ptr[buf_index] = lm_sum[lid][yin];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void integral_sum_rows_image(global const float* buf_ptr, int buf_width,
                                    write_only image2d_t sumImage0, write_only image2d_t sumImage1,
                                    write_only image2d_t sumImage2, write_only image2d_t sumImage3) {
    const int lid = get_local_id(0);
    const int gid = get_group_id(0);
    const int gs = get_global_size(0);
    const int x = get_global_id(0);

    int dst_width = get_image_width(sumImage0);
    int dst_height = get_image_height(sumImage0);

    local float lm_sum[LOCAL_SUM_SIZE][LOCAL_SUM_SIZE];

    for (int xin = x; xin < dst_width; xin += gs) {
        write_imagef(sumImage0, (int2)(xin, 0), 0);

        if ((xin & 1) == 0) {
            write_imagef(sumImage1, (int2)(xin / 2, 0), 0);
        }
        if ((xin & 3) == 0) {
            write_imagef(sumImage2, (int2)(xin / 4, 0), 0);
        }
        if ((xin & 7) == 0) {
            write_imagef(sumImage3, (int2)(xin / 8, 0), 0);
        }
    }

    if (x < dst_height - 1) {
        write_imagef(sumImage0, (int2)(0, x + 1), 0);

        if (((x + 1) & 1) == 0) {
            write_imagef(sumImage1, (int2)(0, (x + 1) / 2), 0);
        }
        if (((x + 1) & 3) == 0) {
            write_imagef(sumImage2, (int2)(0, (x + 1) / 4), 0);
        }
        if (((x + 1) & 7) == 0) {
            write_imagef(sumImage3, (int2)(0, (x + 1) / 8), 0);
        }
    }

    int buf_index = x;
    float accum = 0;
    for (int y = 1; y < dst_width; y += LOCAL_SUM_SIZE) {
#pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, buf_index += buf_width) {
            accum += buf_ptr[buf_index];
            lm_sum[yin][lid] = accum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (y + lid < dst_width) {
            int yin_max = min(dst_height - 1 - LOCAL_SUM_SIZE * gid, LOCAL_SUM_SIZE);
#pragma unroll
            for (int yin = 0; yin < yin_max; yin++) {
                int2 outCoords = (int2)(y + lid, yin + LOCAL_SUM_SIZE * gid + 1);
                write_imagef(sumImage0, outCoords, lm_sum[lid][yin]);

                if (all((outCoords & 1) == 0)) {
                    write_imagef(sumImage1, outCoords / 2, lm_sum[lid][yin]);
                }
                if (all((outCoords & 3) == 0)) {
                    write_imagef(sumImage2, outCoords / 4, lm_sum[lid][yin]);
                }
                if (all((outCoords & 7) == 0)) {
                    write_imagef(sumImage3, outCoords / 8, lm_sum[lid][yin]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

float L2Norm(global const float4* p1, global const float4* p2, int n) {
    float4 sum = 0;
    for (int i = 0; i < n; i++) {
        float4 diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return native_sqrt(sum.x + sum.y + sum.z + sum.w);
}

typedef struct DMatch {
    int queryIdx;  // query descriptor index
    int trainIdx;  // train descriptor index
    float distance;
} DMatch;

#define MATCH_BLOCK_SIZE 24

kernel void matchKeyPoints(global const float4 *descriptor1,
                           int descriptor1_stride,
                           global const float4 *descriptor2,
                           int descriptor2_stride,
                           int descriptor2_height,
                           global DMatch* matchedPoints) {
    local int trainIdx[MATCH_BLOCK_SIZE];
    local float distance[MATCH_BLOCK_SIZE];

    const int lid = get_local_id(1);
    const int i = get_global_id(0);

    trainIdx[lid] = -1;
    distance[lid] = -1;

    barrier(CLK_LOCAL_MEM_FENCE);

    global const float4* p1 = &descriptor1[descriptor1_stride/4 * i];
    float distance_min = 100;
    int j_min = 0;

    for (int j = 0; j < descriptor2_height; j += MATCH_BLOCK_SIZE) {
        if (j + lid >= descriptor2_height) {
            break;
        }
        global const float4* p2 = &descriptor2[descriptor2_stride/4 * (j + lid)];
        float distance_t = L2Norm(p1, p2, 64/4);
        if (distance_t < distance_min) {
            distance_min = distance_t;
            j_min = j + lid;
        }
    }

    trainIdx[lid] = j_min;
    distance[lid] = distance_min;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int j_min = 0;
        float distance_min = 100;
        for (int k = 0; k < MATCH_BLOCK_SIZE; k++) {
            if (distance[k] < distance_min) {
                distance_min = distance[k];
                j_min = trainIdx[k];
            }
        }
        DMatch match = { i, j_min, distance_min };
        matchedPoints[i] = match;
    }
}

typedef struct transform {
    float matrix[3][3];
} transform;

kernel void registerAndFuse(read_only image2d_t inputImage0,
                            read_only image2d_t inputImage1,
                            write_only image2d_t outputImage,
                            transform homography,
                            sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 input_norm = 1.0 / convert_float2(get_image_dim(outputImage));

    float x = imageCoordinates.x;
    float y = imageCoordinates.y;

    float u = homography.matrix[0][0] * x + homography.matrix[0][1] * y + homography.matrix[0][2];
    float v = homography.matrix[1][0] * x + homography.matrix[1][1] * y + homography.matrix[1][2];
    float w = homography.matrix[2][0] * x + homography.matrix[2][1] * y + homography.matrix[2][2];
    float xx = u / w;
    float yy = v / w;

    float4 input0 = read_imagef(inputImage0, imageCoordinates);
    float4 input1 = read_imagef(inputImage1, linear_sampler, ((float2)(xx, yy) + 0.5) * input_norm);

    write_imagef(outputImage, imageCoordinates, (input0 + input1) / 2);
}

kernel void registerImage(read_only image2d_t inputImage,
                          write_only image2d_t outputImage,
                          transform homography,
                          sampler_t linear_sampler) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const float2 input_norm = 1.0 / convert_float2(get_image_dim(outputImage));

    float x = imageCoordinates.x;
    float y = imageCoordinates.y;

    float u = homography.matrix[0][0] * x + homography.matrix[0][1] * y + homography.matrix[0][2];
    float v = homography.matrix[1][0] * x + homography.matrix[1][1] * y + homography.matrix[1][2];
    float w = homography.matrix[2][0] * x + homography.matrix[2][1] * y + homography.matrix[2][2];
    float xx = u / w;
    float yy = v / w;

    float4 input = read_imagef(inputImage, linear_sampler, ((float2)(xx, yy) + 0.5) * input_norm);

    write_imagef(outputImage, imageCoordinates, input);
}
