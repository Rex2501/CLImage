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
#include "gls_geometry.hpp"
#include "homography.hpp"

#define USE_RTL true

#if USE_RTL
#include "rtl/RTL.hpp"
#else
#define USE_SIMPLE_PERSPECTIVE_TRANSFORM false
#define PSEUDO_RANDOM_TEST_SEQUENCE true

#if PSEUDO_RANDOM_TEST_SEQUENCE
#include "PRNG.h"
#endif
#endif

namespace gls {

#if USE_RTL

class HomographyEstimator : public RTL::Estimator<
                                        /*MODEL=*/ gls::Matrix<3, 3>,
                                        /*DATUM=*/ std::pair<Point2f, Point2f>,
                                        /*DATA=*/  std::vector<std::pair<Point2f, Point2f>>> {
public:
    // Calculate the mean of data at the sample indices
    virtual gls::Matrix<3, 3> ComputeModel(const std::vector<std::pair<Point2f, Point2f>>& data, const std::set<int>& samples) {
        assert(samples.size() == 4);

        std::vector<Point2f> selectP1(4);
        std::vector<Point2f> selectP2(4);
        int i = 0;
        for (auto itr = samples.begin(); itr != samples.end(); itr++, i++) {
            const auto& p = data[*itr];
            selectP1[i] = p.first;
            selectP2[i] = p.second;
        }

        try {
            return findHomography(selectP1, selectP2);
        } catch (const std::logic_error& e) {
            printf("Perspective transformation matrix error.");
            return gls::Matrix<3, 3>::identity();
        }
    }

    // Calculate error between the mean and given datum
    virtual float ComputeError(const gls::Matrix<3, 3>& homography, const std::pair<Point2f, Point2f>& datum) {
        const auto p1t = applyHomography(datum.first, homography);
        const auto diff = gls::Vector<2>(p1t - datum.second);
        return dot(diff, diff);
    }
};

#define USE_MLESAC true

gls::Matrix<3, 3> RANSAC(const std::vector<std::pair<Point2f, Point2f>> matchpoints, float threshold, int max_iterations) {
    HomographyEstimator estimator;
#if USE_MLESAC
    RTL::MLESAC<gls::Matrix<3, 3>, std::pair<Point2f, Point2f>, std::vector<std::pair<Point2f, Point2f>> > ransac(&estimator);
    ransac.SetParamSigmaScale(2 * 1.96);
#else
    RTL::RANSAC<gls::Matrix<3, 3>, std::pair<Point2f, Point2f>, std::vector<std::pair<Point2f, Point2f>> > ransac(&estimator);
#endif
    gls::Matrix<3, 3> model;
    ransac.SetParamThreshold(threshold);
    ransac.SetParamIteration(max_iterations);
    const auto loss = ransac.FindBest(model, matchpoints, (int) matchpoints.size(), 4);

    std::cout << "RTL RANSAC loss: " << loss << std::endl;

    // Refine RANSAC projection matrix parameters using the best interior points
    constexpr bool interiorPointsProjection = true;
    if (interiorPointsProjection) {
        const auto inliers = ransac.FindInliers(model, matchpoints, (int) matchpoints.size());

        std::vector<Point2f> p1(inliers.size()), p2(inliers.size());
        for (int i = 0; i < inliers.size(); i++) {
            const auto& p = matchpoints[inliers[i]];
            p1[i] = p.first;
            p2[i] = p.second;
        }
        model = findHomography(p1, p2);
    }
    return model;
}

#else

gls::Matrix<3, 3> RANSAC(const std::vector<std::pair<Point2f, Point2f>> matchpoints, float threshold, int max_iterations) {
    assert(matchpoints.size() > 0);

    // Calculate the maximum set of interior points
    int iters = max_iterations;
    int max_innerP = 0;
    std::vector<int> innerPvInd;  // Inner point set index - temporary
    std::vector<int> innerPvInd_i;
    // generate random table
    auto selectIndex = new int[max_iterations][4];

#if PSEUDO_RANDOM_TEST_SEQUENCE
    const std::array<uint64_t, 16> prng_seed = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    PRNG prng(prng_seed);
#else
    srand((unsigned) time(NULL));  // Use time as seed, each time the random number is different
#endif
    int pCount = (int)matchpoints.size();

    for (int i = 0; i < max_iterations; i++) {
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
        std::vector<Point2f> selectP1(4);
        std::vector<Point2f> selectP2(4);

        for (int i = 0; i < 4; i++) {
            const auto& p = matchpoints[selectIndex[k][i]];
            selectP1[i] = p.first;
            selectP2[i] = p.second;
        }

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
            for (int i = 0; i < matchpoints.size(); i++) {
                const auto& p = matchpoints[i];
                const auto p1t = applyHomography(p.first, homography);
                const auto diff = gls::Vector<2>(p1t - p.second);
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
                const float p = 0.995;
                float ep = (float)(matchpoints.size() - innerP) / matchpoints.size();

                // avoid inf's & nan's
                const float eps = std::numeric_limits<float>::epsilon();

                float num_ = std::max(1.f - p, eps);
                float denom_ = 1. - pow(1.f - ep, 4);

                if (denom_ < eps)
                    iters = 0;
                else {
                    float num = log(num_);
                    float denom = log(denom_);
                    iters = (denom >= 0 || -num >= max_iterations * (-denom) ? max_iterations : (int)(num / denom));
                    // std::cout << "Updated iters to " << iters << std::endl;
                }
            }
            innerPvInd.clear();
        } catch (const std::logic_error& e) {
            printf("Perspective transformation matrix error.");
        }
    }
    delete [] selectIndex;

    printf(" RANSAC interior point ratio - number of loops: %d %ld %d \t\n ", max_innerP, matchpoints.size(), k);

    // Calculate projection matrix parameters based on interior points

    std::vector<Point2f> _p1(max_innerP), _p2(max_innerP);
    for (int i = 0; i < max_innerP; i++) {
        const auto& p = matchpoints[innerPvInd_i[i]];
        _p1[i] = p.first;
        _p2[i] = p.second;
    }

#if USE_SIMPLE_PERSPECTIVE_TRANSFORM
    return getPerspectiveTransformLSM2(_p1, _p2);
#else
    return findHomography(_p1, _p2);
#endif
}

#endif

} // namespace gls
