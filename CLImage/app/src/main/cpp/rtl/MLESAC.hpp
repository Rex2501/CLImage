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

#ifndef __RTL_MLESAC__
#define __RTL_MLESAC__

#include "MSAC.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace RTL {

template <class Model, class Datum, class Data>
class MLESAC : virtual public RANSAC<Model, Datum, Data> {
   public:
    MLESAC(Estimator<Model, Datum, Data>* estimator) : RANSAC<Model, Datum, Data>(estimator) {
        SetParamIterationEM();
        SetParamSigmaScale();
    }

    void SetParamIterationEM(int iteration = 5) { paramIterationEM = iteration; }

    int GetParamIterationEM(void) { return paramIterationEM; }

    void SetParamSigmaScale(float scale = 1.96) { paramSigmaScale = scale; }

    float GetParamSigmaScale(void) { return paramSigmaScale; }

   protected:
    void Initialize(const Data& data, int N) override {
        RANSAC<Model, Datum, Data>::Initialize(data, N);
        dataError2.resize(N);
        dataSigma = RANSAC<Model, Datum, Data>::paramThreshold / paramSigmaScale;
        dataSigma2 = dataSigma * dataSigma;
    }

    float EvaluateModel(const Model& model, const Data& data, int N) override {
        // Calculate squared errors
        float minError = HUGE_VALF, maxError = -HUGE_VALF;
        for (int i = 0; i < N; i++) {
            float error = RANSAC<Model, Datum, Data>::toolEstimator->ComputeError(model, data[i]);
            if (error < minError) minError = error;
            if (error > maxError) maxError = error;
            dataError2[i] = error * error;
        }

        // Estimate the inlier ratio using EM
        const float nu = maxError - minError;
        float gamma = 0.5;
        for (int iter = 0; iter < paramIterationEM; iter++) {
            float sumPosteriorProb = 0;
            const float probOutlier = (1 - gamma) / nu;
            const float probInlierCoeff = gamma / std::sqrt(2 * M_PI * dataSigma2);
            for (int i = 0; i < N; i++) {
                float probInlier = probInlierCoeff * std::exp(-0.5 * dataError2[i] / dataSigma2);
                sumPosteriorProb += probInlier / (probInlier + probOutlier);
            }
            float prev_gamma = gamma;
            gamma = sumPosteriorProb / N;

            // EM convergence check
            if (std::abs(prev_gamma - gamma) < 1.0e-5) {
                break;
            }
        }

        // Evaluate the model
        float sumLogLikelihood = 0;
        const float probOutlier = (1 - gamma) / nu;
        const float probInlierCoeff = gamma / std::sqrt(2 * M_PI * dataSigma2);
        for (int i = 0; i < N; i++) {
            float probInlier = probInlierCoeff * std::exp(-0.5 * dataError2[i] / dataSigma2);
            sumLogLikelihood -= std::log(probInlier + probOutlier);
        }

        // Adaptive Termination
        // TODO: this should ideally live within IsContinued, some refactoring might be necessary
        if (sumLogLikelihood < minSumLogLikelihood) {
            minSumLogLikelihood = sumLogLikelihood;

            const float beta = RANSAC<Model, Datum, Data>::paramThreshold; // Error tolerance - the threshold is probably a good quantifier
            const float k = std::erf(beta / (M_SQRT2 * dataSigma));
            const float denom = std::log(1.0 - std::pow(k, 4) * std::pow(gamma, 4));
            if (std::abs(denom) > FLT_EPSILON) {
                constexpr float successProbability = 0.991;
                const int newMaxIteration = (int)std::floor(std::log(1 - successProbability) / denom);
                if (newMaxIteration < RANSAC<Model, Datum, Data>::paramIteration) {
                    RANSAC<Model, Datum, Data>::paramIteration = newMaxIteration;
                }
            }
        }

        return sumLogLikelihood;
    }

    bool IsContinued(int iteration) override {
        return RANSAC<Model, Datum, Data>::IsContinued(iteration);
    }

    int paramIterationEM;

    float paramSigmaScale;

    std::vector<float> dataError2;

    float dataSigma;
    float dataSigma2;
    float minSumLogLikelihood = FLT_MAX;
};

}  // namespace RTL

#endif  // End of '__RTL_MLESAC__'
