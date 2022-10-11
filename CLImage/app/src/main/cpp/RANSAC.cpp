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

#include "RANSAC.hpp"

#include <float.h>

#include "gls_linalg.hpp"
#include "homography.hpp"

#define USE_SIMPLE_PERSPECTIVE_TRANSFORM false

#define PSEUDO_RANDOM_TEST_SEQUENCE true

#if PSEUDO_RANDOM_TEST_SEQUENCE
#include "PRNG.h"
#endif

namespace gls {

#if USE_SIMPLE_PERSPECTIVE_TRANSFORM
gls::image<float> MatrixMultiply(const gls::image<float>& X1, const gls::image<float>& X2) {
    assert(X1.width == X2.height);

    auto result = gls::image<float>(X2.width, X1.height);

    for (int i = 0; i < result.height; i++) {
        for (int j = 0; j < result.width; j++) {
            float sum = 0;
            for (int ki = 0; ki < X1.width; ki++) {
                sum += X1[i][ki] * X2[ki][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

template <typename TA, typename TB>
void getPerspectiveTransformAB(const std::vector<Point2f>& src, const std::vector<Point2f>& dst, TA& a, TB& b) {
    const int count = (int) src.size();
    assert(a.width == 8 && a.height == 2 * count);
    assert(b.width == 1 && b.height == 2 * count);

    for (int i = 0; i < count; i++) {
        a[i][0] = a[i + count][3] = src[i].x;
        a[i][1] = a[i + count][4] = src[i].y;
        a[i][2] = a[i + count][5] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i + count][0] = a[i + count][1] = a[i + count][2] = 0;
        a[i][6] = -src[i].x * dst[i].x;
        a[i][7] = -src[i].y * dst[i].x;
        a[i + count][6] = -src[i].x * dst[i].y;
        a[i + count][7] = -src[i].y * dst[i].y;
        b[i][0] = dst[i].x;
        b[i + count][0] = dst[i].y;
    }
}

gls::Matrix<3, 3> getPerspectiveTransformIata(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    assert(src.size() == 4 && dst.size() == 4);

    gls::Matrix<8, 8> a;
    gls::Matrix<8, 1> b;
    getPerspectiveTransformAB(src, dst, a, b);

    gls::Matrix<8, 8> at = transpose(a);
    gls::Matrix<8, 8> ata = at * a;
    gls::Vector<8> atb = at * b;

    const auto h = LUSolve(ata, atb);

    return {
        h[0], h[1], h[2],
        h[3], h[4], h[5],
        h[6], h[7], 1
    };
}

template <typename T>
gls::image<T> transpose(const gls::image<T>& a) {
    auto at = gls::image<float>(a.height, a.width);
    for (int j = 0; j < a.height; j++) {
        for (int i = 0; i < a.width; i++) {  // find the transpose of a
            at[i][j] = a[j][i];
        }
    }
    return at;
}

gls::Matrix<3, 3> getPerspectiveTransformLSM2(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    int count = (int) src.size();

    auto a = gls::image<float>(8, 2 * count);
    auto b = gls::image<float>(1, 2 * count);

    getPerspectiveTransformAB(src, dst, a, b);

    /* Normal matrix to find least squares */
    const auto at = transpose(a);
    const auto ata = MatrixMultiply(at, a);
    const auto atb = MatrixMultiply(at, b);

    gls::Matrix<8, 8> aa;
    gls::Vector<8> bb;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            aa[i][j] = ata[i][j];
        }
    }
    for (int i = 0; i < 8; i++) {
        bb[i] = atb[i][0];
    }

    // TODO: replace this with a proper solver
    // return inverse(aa) * bb;
    const auto h = LUSolve(aa, bb);

    return {
        h[0], h[1], h[2],
        h[3], h[4], h[5],
        h[6], h[7], 1
    };
}
#endif

Point2f transform(const Point2f& p, const gls::Matrix<3, 3>& homography) {
    gls::Vector<3> pv = { p.x, p.y, 1 };
    const auto op = homography * pv;            // Transformed point in homogeneous coordinates
    return { op[0] / op[2], op[1] / op[2] };    // Converted to cartesian coordinates
}

gls::Matrix<3, 3> getRANSAC2(const std::vector<Point2f>& p1, const std::vector<Point2f>& p2, float threshold, int count) {
    assert(p1.size() > 0 && p2.size() > 0);

    // Calculate the maximum set of interior points
    int max_iters = 2000;
    int iters = max_iters;
    int max_innerP = 0;
    std::vector<int> innerPvInd;  // Inner point set index - temporary
    std::vector<int> innerPvInd_i;
    // generate random table
    int selectIndex[2000][4];

#if PSEUDO_RANDOM_TEST_SEQUENCE
    const std::array<uint64_t, 16> prng_seed = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    PRNG prng(prng_seed);
#else
    srand((unsigned) time(NULL));  // Use time as seed, each time the random number is different
#endif
    int pCount = (int)p1.size();

    for (int i = 0; i < 2000; i++) {
        for (int j = 0; j < 4; j++) {
            int ii = 0;
            int temp = 0;
            selectIndex[i][0] = selectIndex[i][1] = selectIndex[i][2] = selectIndex[i][3] =
            pCount + 1;
            while (ii < 4) {
#if PSEUDO_RANDOM_TEST_SEQUENCE
                temp = prng.getRandomInt(0, RAND_MAX) % pCount;
#else
                temp = rand() % pCount;
#endif
                if (temp != selectIndex[i][0] && temp != selectIndex[i][1] &&
                    temp != selectIndex[i][2] && temp != selectIndex[i][3]) {
                    selectIndex[i][ii] = temp;
                    ii++;
                }
            }
        }
    }

    int k = 0;
    for (; k < iters; k++) {
        const std::vector<Point2f> selectP1 = {
            p1[selectIndex[k][0]],
            p1[selectIndex[k][1]],
            p1[selectIndex[k][2]],
            p1[selectIndex[k][3]]
        };
        const std::vector<Point2f> selectP2 = {
            p2[selectIndex[k][0]],
            p2[selectIndex[k][1]],
            p2[selectIndex[k][2]],
            p2[selectIndex[k][3]]
        };

        // Calculate the perspective transformation matrix
        // (currently the singular matrix inverse cannot be solved, and the
        // singular decomposition inversion method will be improved in the future)
        try {
#if USE_SIMPLE_PERSPECTIVE_TRANSFORM
            const auto homography = getPerspectiveTransformIata(selectP1, selectP2);
#else
            const auto homography = findHomography(selectP1, selectP2);
#endif
            // Calculate the model parameter error, if the error is greater than the threshold, discard
            // this set of model parameters
            int innerP = 0;
            for (int i = 0; i < p1.size(); i++) {
                const auto p1t = transform(p1[i], homography);
                const auto diff = gls::Vector<2>(p1t) - gls::Vector<2>(p2[i]);
                const auto errSquare = dot(diff, diff);
                if (errSquare < threshold) {
                    innerP++;
                    innerPvInd.push_back(i);
                }
            }
            if (innerP > max_innerP) {
                max_innerP = innerP;
                innerPvInd_i = innerPvInd;
                // update the number of iterations
                float p = 0.995;
                float ep = (float)(p1.size() - innerP) / p1.size();
                // avoid inf's & nan's
                float num_ = std::max(1.f - p, FLT_MIN);
                float denom_ = 1. - pow(1.f - ep, 4);

                if (denom_ < FLT_MIN)
                    iters = 0;
                else {
                    float num = log(num_);
                    float denom = log(denom_);
                    iters =
                    (denom >= 0 || -num >= max_iters * (-denom) ? max_iters : (int)(num / denom));
                }
            }
            innerPvInd.clear();
        } catch (const std::logic_error& e) {
            printf("Perspective transformation matrix transformation error");
        }
    }
    printf(" RANSAC interior point ratio - number of loops: %d %ld %d \t\n ", max_innerP, p1.size(), k);

    // Calculate projection matrix parameters based on interior points

    std::vector<Point2f> _p1(max_innerP), _p2(max_innerP);
    for (int i = 0; i < max_innerP; i++) {
        _p1[i] = p1[innerPvInd_i[i]];
        _p2[i] = p2[innerPvInd_i[i]];
    }

#if USE_SIMPLE_PERSPECTIVE_TRANSFORM
    return getPerspectiveTransformLSM2(_p1, _p2);
#else
    return findHomography(_p1, _p2);
#endif
}

} // namespace gls
