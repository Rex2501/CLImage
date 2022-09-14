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

typedef struct SurfHF {
    int2 p0, p1, p2, p3;
    float w;
} SurfHF;

float calcHaarPattern(read_only image2d_t inputImage, const int2 p, constant SurfHF f[], int N) {
    float d = 0;
    for (int k = 0; k < N; k++) {
        constant SurfHF* fk = &f[k];
        d += (read_imagef(inputImage, p + fk->p0).x +
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
                            constant SurfHF Dx[3],
                            constant SurfHF Dy[3],
                            constant SurfHF Dxy[4]) {
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

const constant int KeyPointMaxima_MaxCount = 20000;

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

kernel void findMaximaInLayer(read_only image2d_t sumImage, read_only image2d_t detImage0,
                              read_only image2d_t detImage1, read_only image2d_t detImage2,
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

kernel void rowIntegral(global float* src, global float* dst, local float* temp, int w) {
    int lid = get_local_id(0);
    int wid = get_group_id(0);
    int n = get_local_size(0) * 2;

    int chunks = (w + n - 1) / n;

    for (int c = 0; c < chunks; c++) {
        int offset = 1;

        if (c > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        temp[2 * lid] = (c * n + 2 * lid < w) ? src[c * n + 2 * lid + wid * w] : 0;
        temp[2 * lid + 1] = (c * n + 2 * lid + 1 < w) ? src[c * n + 2 * lid + 1 + wid * w] : 0;

        for (int d = n >> 1; d > 0; d >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < d) {
                int ai = offset * (2 * lid + 1) - 1;
                int bi = offset * (2 * lid + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset *= 2;
        }

        if (lid == 0) {
            temp[n - 1] = 0.f;
        }

        // down-sweep
        for (int d = 1; d < n; d *= 2) {
            offset >>= 1;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < d) {
                int ai = offset * (2 * lid + 1) - 1;
                int bi = offset * (2 * lid + 2) - 1;

                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (c * n + 2 * lid < w) {
            dst[c * n + 2 * lid + wid * w] = dst[c * n - 2 + wid * w] + temp[2 * lid] + src[c * n + 2 * lid + wid * w];
        }
        if (c * n + 2 * lid + 1 < w) {
            dst[c * n + 2 * lid + 1 + wid * w] = dst[c * n - 1 + wid * w] + temp[2 * lid + 1] + src[c * n + 2 * lid + 1 + wid * w];
        }
    }
}

kernel void colIntegral(global float* src, global float* dst, __local float* temp, int h, int w) {
    int lid = get_local_id(0);
    int wid = get_group_id(0);
    int n = get_local_size(0) * 2;

    int chunks = (w + n - 1) / n;

    for (int c = 0; c < chunks; c++) {
        int offset = 1;

        if (c > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        temp[2 * lid] = (c * n + 2 * lid < h) ? src[(c * n + 2 * lid) * w + wid] : 0;
        temp[2 * lid + 1] = (c * n + 2 * lid + 1 < h) ? src[(c * n + 2 * lid + 1) * w + wid] : 0;

        for (int d = n >> 1; d > 0; d >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < d) {
                int ai = offset * (2 * lid + 1) - 1;
                int bi = offset * (2 * lid + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset *= 2;
        }

        if (lid == 0) {
            temp[n - 1] = 0;
        }

        // down-sweep
        for (int d = 1; d < n; d *= 2) {
            offset >>= 1;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < d) {
                int ai = offset * (2 * lid + 1) - 1;
                int bi = offset * (2 * lid + 2) - 1;

                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (c * n + 2 * lid < h) {
            dst[(c * n + 2 * lid) * w + wid] = dst[(c * n - 2) * w + wid] + temp[2 * lid] + src[2 * lid * w + wid];
        }
        if (c * n + 2 * lid + 1 < h) {
            dst[(c * n + 2 * lid + 1) * w + wid] = dst[(c * n - 1) * w + wid] + temp[2 * lid + 1] + src[(2 * lid + 1) * w + wid];
        }
    }
}

#define LOCAL_SUM_SIZE      16
#define LOCAL_SUM_STRIDE    (LOCAL_SUM_SIZE + 1)


kernel void integral_sum_cols(__global const float *src_ptr, int src_step, int src_offset, int rows, int cols,
                              __global uchar *buf_ptr, int buf_step, int buf_offset)
{
    __local float lm_sum[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    int x = get_global_id(0);
    int src_index = x + src_offset;

    float accum = 0;
    for (int y = 0; y < rows; y += LOCAL_SUM_SIZE)
    {
        int lsum_index = lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, src_index+=src_step, lsum_index += LOCAL_SUM_STRIDE)
        {
            if ((x < cols) && (y + yin < rows))
            {
                __global const float *src = src_ptr + src_index;
                accum += src[0];
            }
            lm_sum[lsum_index] = accum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //int buf_index = buf_offset + buf_step * LOCAL_SUM_COLS * gid + sizeof(float) * y + sizeof(float) * lid;
        int buf_index = mad24(buf_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(float), y + lid, buf_offset));

        lsum_index = LOCAL_SUM_STRIDE * lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, lsum_index ++)
        {
            __global float *buf = (__global float *)(buf_ptr + buf_index);
            buf[0] = lm_sum[lsum_index];
            buf_index += buf_step;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void integral_sum_rows(__global const uchar *buf_ptr, int buf_step, int buf_offset,
                              __global uchar *dst_ptr, int dst_step, int dst_offset, int rows, int cols)
{
    __local float lm_sum[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    int gs = get_global_size(0);

    int x = get_global_id(0);

    __global float *dst = (__global float *)(dst_ptr + dst_offset);
    for (int xin = x; xin < cols; xin += gs)
    {
        dst[xin] = 0;
    }
    dst_offset += dst_step;

    if (x < rows - 1)
    {
        dst = (__global float *)(dst_ptr + mad24(x, dst_step, dst_offset));
        dst[0] = 0;
    }

    int buf_index = mad24((int)sizeof(float), x, buf_offset);
    float accum = 0;

    for (int y = 1; y < cols; y += LOCAL_SUM_SIZE)
    {
        int lsum_index = lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, lsum_index += LOCAL_SUM_STRIDE)
        {
            __global const float *buf = (__global const float *)(buf_ptr + buf_index);
            accum += buf[0];
            lm_sum[lsum_index] = accum;
            buf_index += buf_step;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (y + lid < cols)
        {
            //int dst_index = dst_offset + dst_step *  LOCAL_SUM_COLS * gid + sizeof(float) * y + sizeof(float) * lid;
            int dst_index = mad24(dst_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(float), y + lid, dst_offset));
            lsum_index = LOCAL_SUM_STRIDE * lid;
            int yin_max = min(rows - 1 -  LOCAL_SUM_SIZE * gid, LOCAL_SUM_SIZE);
            #pragma unroll
            for (int yin = 0; yin < yin_max; yin++, lsum_index++)
            {
                dst = (__global float *)(dst_ptr + dst_index);
                dst[0] = lm_sum[lsum_index];
                dst_index += dst_step;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
