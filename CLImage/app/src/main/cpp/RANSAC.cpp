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

#define USE_SIMPLE_PERSPECTIVE_TRANSFORM false

#define PSEUDO_RANDOM_TEST_SEQUENCE true

#if PSEUDO_RANDOM_TEST_SEQUENCE
#include "PRNG.h"
#endif

//#define USE_GRANSAC true
#define USE_RTL true

#if USE_GRANSAC
#include "GRANSAC/GRANSAC.hpp"
#elif USE_RTL
#include "rtl/RTL.hpp"
#endif

namespace gls {

// --- GRANSAC ---

#if USE_GRANSAC
typedef gls::basic_point<GRANSAC::VPFloat> VPPoint;

struct MatchedPoints : public GRANSAC::AbstractParameter {
    VPPoint p1, p2;

    MatchedPoints(const VPPoint& _p1, const VPPoint& _p2) : p1(_p1), p2(_p2) { }
};

class HomographyModel : public GRANSAC::AbstractModel<4> {
   protected:
    std::array<std::shared_ptr<GRANSAC::AbstractParameter>, 4> m_MinModelParams;

    gls::Matrix<3, 3> homography;

    GRANSAC::VPFloat ComputeDistanceMeasure(std::shared_ptr<GRANSAC::AbstractParameter> Param) override {
        auto pair = std::dynamic_pointer_cast<MatchedPoints>(Param);
        if (pair == nullptr)
            throw std::runtime_error(
                "HomographyModel::ComputeDistanceMeasure() - Passed parameter are not instances of MatchedPoints.");

        const auto p1t = applyHomography(Point2f(pair->p1), homography);
        const auto diff = gls::Vector<2>(p1t) - gls::Vector<2>(pair->p2);
        return dot(diff, diff);  // error squared
    }

   public:
    HomographyModel(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> &InputParams) {
        Initialize(InputParams);
    };

    void Initialize(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>& InputParams) override {
        if (InputParams.size() != 4)
            throw std::runtime_error("HomographyModel - Number of input parameters does not match minimum number required for this model.");

        // Check for AbstractParamter types
        auto pair1 = std::dynamic_pointer_cast<MatchedPoints>(InputParams[0]);
        auto pair2 = std::dynamic_pointer_cast<MatchedPoints>(InputParams[1]);
        auto pair3 = std::dynamic_pointer_cast<MatchedPoints>(InputParams[2]);
        auto pair4 = std::dynamic_pointer_cast<MatchedPoints>(InputParams[3]);

        if (pair1 == nullptr || pair2 == nullptr || pair3 == nullptr || pair4 == nullptr)
            throw std::runtime_error("HomographyModel - InputParams type mismatch. It is not an instance of MatchedPoints.");

        std::copy(InputParams.begin(), InputParams.end(), m_MinModelParams.begin());

        const std::vector<Point2f> selectP1 = {
            pair1->p1,
            pair2->p1,
            pair3->p1,
            pair4->p1
        };
        const std::vector<Point2f> selectP2 = {
            pair1->p2,
            pair2->p2,
            pair3->p2,
            pair4->p2
        };

        // Calculate the perspective transformation matrix
        try {
            homography = findHomography(selectP1, selectP2);
        } catch (const std::logic_error& e) {
            printf("Perspective transformation matrix transformation error");
        }
    }

    std::pair<GRANSAC::VPFloat, std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>> Evaluate(
        const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>& EvaluateParams,
        GRANSAC::VPFloat Threshold) override {
            std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> Inliers;
            int nTotalParams = (int) EvaluateParams.size();
            int nInliers = 0;

            for (auto& Param : EvaluateParams) {
                if (ComputeDistanceMeasure(Param) < Threshold) {
                    Inliers.push_back(Param);
                    nInliers++;
                }
            }

            GRANSAC::VPFloat InlierFraction = GRANSAC::VPFloat(nInliers) / GRANSAC::VPFloat(nTotalParams); // This is the inlier fraction

            return std::make_pair(InlierFraction, Inliers);
    }
};


gls::Matrix<3, 3> RANSAC(const std::vector<Point2f>& matchpoints1, const std::vector<Point2f>& matchpoints2, float threshold, int max_iterations) {
    gls::Matrix<3, 3> homography;

    const auto nPoints = matchpoints1.size();

    std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> matches(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        matches[i] = std::make_shared<MatchedPoints>(MatchedPoints { VPPoint(matchpoints1[i]), VPPoint(matchpoints2[i]) });
    }

    GRANSAC::RANSAC<HomographyModel, 4> Estimator;
    Estimator.Initialize(threshold, max_iterations); // Threshold, iterations
    Estimator.Estimate(matches);

    auto bestInliers = Estimator.GetBestInliers();
    const int inliersCount = (int) bestInliers.size();
    if (inliersCount > 0) {
        std::vector<Point2f> _p1(inliersCount), _p2(inliersCount);

        for (int i = 0; i < inliersCount; i++) {
            auto RPt = std::dynamic_pointer_cast<MatchedPoints>(bestInliers[i]);

            _p1[i] = RPt->p1;
            _p2[i] = RPt->p2;
        }

        homography = findHomography(_p1, _p2);

        std::cout << "GRANSAC found " << inliersCount << " inliers, homography:\n" << homography << std::endl;
    }

    return homography;
}

#elif USE_RTL
// --- RTL ---

struct MatchedPoints {
    Point2f p1, p2;
};

// A mean calculator
class HomographyEstimator : public RTL::Estimator<gls::Matrix<3, 3>, MatchedPoints, std::vector<MatchedPoints>> {
public:
    // Calculate the mean of data at the sample indices
    virtual gls::Matrix<3, 3> ComputeModel(const std::vector<MatchedPoints>& data, const std::set<int>& samples) {
        assert(samples.size() == 4);

        std::vector<Point2f> selectP1(4);
        std::vector<Point2f> selectP2(4);
        int i = 0;
        for (auto itr = samples.begin(); itr != samples.end(); itr++, i++) {
            selectP1[i] = data[*itr].p1;
            selectP2[i] = data[*itr].p2;
        }

        try {
            return findHomography(selectP1, selectP2);
        } catch (const std::logic_error& e) {
            printf("Perspective transformation matrix transformation error");
            return gls::Matrix<3, 3>::identity();
        }
    }

    // Calculate error between the mean and given datum
    virtual double ComputeError(const gls::Matrix<3, 3>& homography, const MatchedPoints& datum) {
        const auto p1t = applyHomography(datum.p1, homography);
        const auto diff = gls::Vector<2>(p1t) - gls::Vector<2>(datum.p2);
        return dot(diff, diff);
    }
};

gls::Matrix<3, 3> RANSAC(const std::vector<Point2f>& matchpoints1, const std::vector<Point2f>& matchpoints2, float threshold, int max_iterations) {
    const auto nPoints = matchpoints1.size();

    std::vector<MatchedPoints> matches(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        matches[i] = MatchedPoints { matchpoints1[i], matchpoints2[i] };
    }

    HomographyEstimator estimator;
    RTL::RANSAC<gls::Matrix<3, 3>, MatchedPoints, std::vector<MatchedPoints> > ransac(&estimator);
    gls::Matrix<3, 3> model;
    ransac.SetParamThreshold(threshold);
    ransac.SetParamIteration(max_iterations);
    double loss = ransac.FindBest(model, matches, (int) matches.size(), 4);

    std::cout << "RTL RANSAC loss: " << loss << std::endl;

    return model;
}

#else
// --- ORIGINAL ---

gls::Matrix<3, 3> RANSAC(const std::vector<Point2f>& p1, const std::vector<Point2f>& p2, float threshold, int max_iterations) {
    assert(p1.size() > 0 && p2.size() > 0);

    // Calculate the maximum set of interior points
    int iters = max_iterations;
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
                const float p = 0.995;
                float ep = (float)(p1.size() - innerP) / p1.size();

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

#endif

} // namespace gls
