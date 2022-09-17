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

static const constant float SURF_INTEGRAL_BIAS  = 255;

typedef struct SurfHF {
    int2 p0, p1, p2, p3;
    float w;
} SurfHF;

float calcHaarPattern(read_only image2d_t inputImage, const int2 p, constant SurfHF f[], int N) {
    float d = 0;
    for (int k = 0; k < N; k++) {
        constant SurfHF* fk = &f[k];
        d += SURF_INTEGRAL_BIAS * (read_imagef(inputImage, p + fk->p0).x +
                                   read_imagef(inputImage, p + fk->p3).x -
                                   read_imagef(inputImage, p + fk->p1).x -
                                   read_imagef(inputImage, p + fk->p2).x) * fk->w;
    }
    return d;
}

kernel void calcDetAndTrace(read_only image2d_t sumImage,
                            write_only image2d_t detImage,
                            write_only image2d_t traceImage,
                            int2 margin,
                            int sampleStep,
                            constant SurfHF Dx[3]
#ifndef __APPLE__
                            __attribute__ ((max_constant_size(3 * sizeof(SurfHF))))
#endif
                            ,
                            constant SurfHF Dy[3]
#ifndef __APPLE__
                            __attribute__ ((max_constant_size(3 * sizeof(SurfHF))))
#endif
                            ,
                            constant SurfHF Dxy[4]
#ifndef __APPLE__
                            __attribute__ ((max_constant_size(4 * sizeof(SurfHF))))
#endif
                            ) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const int2 p = imageCoordinates * sampleStep;

    const float dx = calcHaarPattern(sumImage, p, Dx, 3);
    const float dy = calcHaarPattern(sumImage, p, Dy, 3);
    const float dxy = calcHaarPattern(sumImage, p, Dxy, 4);

    write_imagef(detImage, imageCoordinates + margin, dx * dy - 0.81f * dxy * dxy);
    write_imagef(traceImage, imageCoordinates + margin, dx + dy);
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

#define KeyPointMaxima_MaxCount 20000

typedef struct KeyPointMaxima {
    int count;
    KeyPoint keyPoints[KeyPointMaxima_MaxCount];
} KeyPointMaxima;

inline bool solve3x3(const float3 A[3], const float b[3], float x[3]) {
    float det = A[0].x * (A[1].y * A[2].z - A[1].z * A[2].y) -
                A[0].y * (A[1].x * A[2].z - A[1].z * A[2].x) +
                A[0].z * (A[1].x * A[2].y - A[1].y * A[2].x);

    if (det != 0) {
        float invdet = 1.0f / det;
        x[0] = invdet *
               (b[0]   * (A[1].y * A[2].z - A[1].z * A[2].y) -
                A[0].y * (b[1]   * A[2].z - A[1].z * b[2]  ) +
                A[0].z * (b[1]   * A[2].y - A[1].y * b[2]  ));

        x[1] = invdet *
               (A[0].x * (b[1]   * A[2].z - A[1].z * b[2]  ) -
                b[0]   * (A[1].x * A[2].z - A[1].z * A[2].x) +
                A[0].z * (A[1].x * b[2]   - b[1]   * A[2].x));

        x[2] = invdet *
               (A[0].x * (A[1].y * b[2]   - b[1]   * A[2].y) -
                A[0].y * (A[1].x * b[2]   - b[1]   * A[2].x) +
                b[0]   * (A[1].x * A[2].y - A[1].y * A[2].x));

        return true;
    }
    return false;
}

#define N9(_idx, _off_y, _off_x) read_imagef(detImage ## _idx, p + (int2) (_off_x, _off_y)).x

bool interpolateKeypoint(read_only image2d_t detImage0, read_only image2d_t detImage1, read_only image2d_t detImage2,
                         int2 p, int dx, int dy, int ds, KeyPoint* kpt) {
    float B[3] = {
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
    float x[3];
    bool ok = solve3x3(A, B, x);
    ok = ok && (x[0] != 0 || x[1] != 0 || x[2] != 0) &&
         fabs(x[0]) <= 1 && fabs(x[1]) <= 1 && fabs(x[2]) <= 1;

    if (ok) {
        kpt->pt.x += x[0] * dx;
        kpt->pt.y += x[1] * dy;
        kpt->size = (float) rint(kpt->size + x[2] * ds);
    }
    return ok;
}

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

#define LOCAL_SUM_SIZE      16
#define LOCAL_SUM_STRIDE    (LOCAL_SUM_SIZE + 1)

kernel void integral_sum_cols(global const float *src_ptr, int src_width, int src_height,
                              global float *buf_ptr, int buf_width) {
    const int lid = get_local_id(0);
    const int gid = get_group_id(0);
    const int x = get_global_id(0);

    local float lm_sum[LOCAL_SUM_STRIDE][LOCAL_SUM_SIZE];

    int src_index = x;
    float accum = 0;
    for (int y = 0; y < src_height; y += LOCAL_SUM_SIZE) {
#pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, src_index += src_width) {
            if ((x < src_width) && (y + yin < src_height)) {
                accum += src_ptr[src_index] / SURF_INTEGRAL_BIAS;
            }
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

kernel void integral_sum_cols_image(read_only image2d_t sourceImage, global float *buf_ptr, int buf_width) {
    const int lid = get_local_id(0);
    const int gid = get_group_id(0);
    const int x = get_global_id(0);

    local float lm_sum[LOCAL_SUM_STRIDE][LOCAL_SUM_SIZE];

    float accum = 0;
    for (int y = 0; y < get_image_height(sourceImage); y += LOCAL_SUM_SIZE) {
#pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++) {
            accum += read_imagef(sourceImage, (int2)(x, y + yin)).x / SURF_INTEGRAL_BIAS;
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

kernel void integral_sum_rows(global const float *buf_ptr, int buf_width,
                              global float *dst_ptr, int dst_width, int dst_height) {
    const int lid = get_local_id(0);
    const int gid = get_group_id(0);
    const int gs = get_global_size(0);
    const int x = get_global_id(0);

    local float lm_sum[LOCAL_SUM_STRIDE][LOCAL_SUM_SIZE];

    for (int xin = x; xin < dst_width; xin += gs) {
        dst_ptr[xin] = 0;
    }

    if (x < dst_height - 1) {
        dst_ptr[(x + 1) * dst_width] = 0;
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
            int dst_index = dst_width * (LOCAL_SUM_SIZE * gid + 1) + (y + lid);
            int yin_max = min(dst_height - 1 - LOCAL_SUM_SIZE * gid, LOCAL_SUM_SIZE);
#pragma unroll
            for (int yin = 0; yin < yin_max; yin++, dst_index += dst_width) {
                dst_ptr[dst_index] = lm_sum[lid][yin];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void integral_sum_rows_image(global const float* buf_ptr, int buf_width, write_only image2d_t dstImage) {
    const int lid = get_local_id(0);
    const int gid = get_group_id(0);
    const int gs = get_global_size(0);
    const int x = get_global_id(0);

    int dst_width = get_image_width(dstImage);
    int dst_height = get_image_height(dstImage);

    local float lm_sum[LOCAL_SUM_STRIDE][LOCAL_SUM_SIZE];

    for (int xin = x; xin < dst_width; xin += gs) {
        write_imagef(dstImage, (int2)(xin, 0), 0);
    }

    if (x < dst_height - 1) {
        write_imagef(dstImage, (int2)(0, x + 1), 0);
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
                write_imagef(dstImage, (int2)(y + lid, yin + LOCAL_SUM_SIZE * gid + 1), lm_sum[lid][yin]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
