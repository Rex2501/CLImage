// Copyright (c) 2007, Sunglok Choi
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.

#ifndef __RTL_RANSAC__
#define __RTL_RANSAC__

#include <cassert>
#include <cmath>
#include <random>

#include "Base.hpp"

namespace RTL {

template <class Model, class Datum, class Data>
class RANSAC {
   public:
    RANSAC(Estimator<Model, Datum, Data>* estimator) {
        assert(estimator != NULL);

        toolEstimator = estimator;
        SetParamIteration();
        SetParamThreshold();
    }

    virtual float FindBest(Model& best, const Data& data, int N, int M) {
        assert(N > 0 && M > 0);

        Initialize(data, N);

        // Run RANSAC
        float bestloss = HUGE_VAL;
        int iteration = 0;
        while (IsContinued(iteration)) {
            iteration++;

            // 1. Generate hypotheses
            Model model = GenerateModel(data, M);

            // 2. Evaluate the hypotheses
            float loss = EvaluateModel(model, data, N);
            if (loss < bestloss)
                if (!UpdateBest(best, bestloss, model, loss)) goto RANSAC_FIND_BEST_EXIT;
        }

    RANSAC_FIND_BEST_EXIT:
        Terminate(best, data, N);
        return bestloss;
    }

    virtual std::vector<int> FindInliers(const Model& model, const Data& data, int N) {
        std::vector<int> inliers;
        for (int i = 0; i < N; i++) {
            float error = toolEstimator->ComputeError(model, data[i]);
            if (std::abs(error) < paramThreshold) inliers.push_back(i);
        }
        return inliers;
    }

    void SetParamIteration(int iteration = 100) { paramIteration = iteration; }

    int GetParamIteration(void) { return paramIteration; }

    void SetParamThreshold(float threshold = 1) { paramThreshold = threshold; }

    int GetParamThreshold(void) { return paramThreshold; }

   protected:
    virtual bool IsContinued(int iteration) { return (iteration < paramIteration); }

    virtual Model GenerateModel(const Data& data, int M) {
        std::set<int> samples;
        while (static_cast<int>(samples.size()) < M) samples.insert(toolUniform(toolGenerator));
        return toolEstimator->ComputeModel(data, samples);
    }

    virtual float EvaluateModel(const Model& model, const Data& data, int N) {
        float loss = 0;
        for (int i = 0; i < N; i++) {
            float error = toolEstimator->ComputeError(model, data[i]);
            loss += (std::abs(error) > paramThreshold);
        }
        return loss;
    }

    virtual bool UpdateBest(Model& bestModel, float& bestCost, const Model& model, float cost) {
        bestModel = model;
        bestCost = cost;
        return true;
    }

    virtual void Initialize(const Data& data, int N) {
        toolUniform = std::uniform_int_distribution<int>(0, N - 1);
    }

    virtual void Terminate(const Model& bestModel, const Data& data, int N) {}

    std::mt19937 toolGenerator;

    std::uniform_int_distribution<int> toolUniform;

    Estimator<Model, Datum, Data>* toolEstimator;

    int paramSampleSize;

    int paramIteration;

    float paramThreshold;
};  // End of 'RANSAC'

}  // namespace RTL

#endif  // End of '__RTL_RANSAC__'
