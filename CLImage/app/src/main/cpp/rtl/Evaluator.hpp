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

#ifndef __RTL_EVALUATOR__
#define __RTL_EVALUATOR__

#include <algorithm>
#include <chrono>

#include "Base.hpp"

class Score {
   public:
    Score() : tp(0), fp(0), tn(0), fn(0) {}

    int tp;
    int fp;
    int tn;
    int fn;
};

template <class Model, class Datum, class Data>
class Evaluator {
   public:
    Evaluator(RTL::Estimator<Model, Datum, Data>* estimator) {
        assert(estimator != NULL);
        toolEstimator = estimator;
        trueSSE = -1;
    }

    bool SetGroundTruth(const Model& model, const Data& data, int N,
                        const std::vector<int> inliers) {
        trueModel = model;
        trueInliers = inliers;
        noisyData.clear();
        for (int i = 0; i < N; i++) noisyData.push_back(data[i]);

        trueSSE = 0;
        for (size_t i = 0; i < trueInliers.size(); i++) {
            float error = toolEstimator->ComputeError(trueModel, noisyData[trueInliers[i]]);
            trueSSE += error * error;
        }
        return true;
    }

    float EvaluateModel(const Model& model) {
        if (trueSSE <= 0) return -1;  // Check initialization

        float SSE = 0;
        for (size_t i = 0; i < trueInliers.size(); i++) {
            float error = toolEstimator->ComputeError(model, noisyData[trueInliers[i]]);
            SSE += error * error;
        }
        return (SSE / trueSSE);
    }

    Score EvaluateInliers(const std::vector<int>& inliers) {
        Score score;
        if (trueInliers.empty()) return score;  // Check initialization

        for (size_t i = 0; i < inliers.size(); i++) {
            bool found = (std::find(trueInliers.begin(), trueInliers.end(), inliers[i]) !=
                          trueInliers.end());
            if (found)
                score.tp++;
            else
                score.fp++;
        }
        int t = static_cast<int>(trueInliers.size());
        int f = static_cast<int>(noisyData.size() - trueInliers.size());
        score.fn = t - score.tp;
        score.tn = f - score.fp;
        return score;
    }

   protected:
    RTL::Estimator<Model, Datum, Data>* toolEstimator;

    Model trueModel;

    std::vector<int> trueInliers;

    float trueSSE;

    std::vector<Datum> noisyData;
};  // End of 'Evaluator'

class StopWatch {
   public:
    StopWatch() { Start(); }

    bool Start(void) {
        start = std::chrono::high_resolution_clock::now();
        return true;
    }

    double GetElapse(void) {
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        return static_cast<double>(time_span.count());
    }

   private:
    std::chrono::high_resolution_clock::time_point start;
};

#endif  // End of '__RTL_EVALUATOR__'
