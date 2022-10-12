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

gls::Matrix<3, 3> RANSAC(const std::vector<Point2f>& p1, const std::vector<Point2f>& p2, float threshold, int count) {
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
        try {
#if USE_SIMPLE_PERSPECTIVE_TRANSFORM
            const auto homography = getPerspectiveTransformLSM2(selectP1, selectP2);
#else
            const auto homography = findHomography(selectP1, selectP2);
#endif
            // Calculate the model parameter error, if the error is greater than the threshold, discard
            // this set of model parameters
            int innerP = 0;
            for (int i = 0; i < p1.size(); i++) {
                const auto p1t = applyHomography(p1[i], homography);
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
