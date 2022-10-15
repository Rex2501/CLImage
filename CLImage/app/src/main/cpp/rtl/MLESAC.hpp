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
        dataError2 = NULL;
        SetParamIterationEM();
        SetParamSigmaScale();
    }

    void SetParamIterationEM(int iteration = 5) { paramIterationEM = iteration; }

    int GetParamIterationEM(void) { return paramIterationEM; }

    void SetParamSigmaScale(float scale = 1.96) { paramSigmaScale = scale; }

    float GetParamSigmaScale(void) { return paramSigmaScale; }

   protected:
    virtual void Initialize(const Data& data, int N) {
        RANSAC<Model, Datum, Data>::Initialize(data, N);
        dataError2 = new float[N];
        assert(dataError2 != NULL);
        float sigma = RANSAC<Model, Datum, Data>::paramThreshold / paramSigmaScale;
        dataSigma2 = sigma * sigma;
    }

    virtual float EvaluateModel(const Model& model, const Data& data, int N) {
        // Calculate squared errors
        float minError = HUGE_VAL, maxError = -HUGE_VAL;
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
            gamma = sumPosteriorProb / N;
        }

        // Evaluate the model
        float sumLogLikelihood = 0;
        const float probOutlier = (1 - gamma) / nu;
        const float probInlierCoeff = gamma / std::sqrt(2 * M_PI * dataSigma2);
        for (int i = 0; i < N; i++) {
            float probInlier = probInlierCoeff * std::exp(-0.5 * dataError2[i] / dataSigma2);
            sumLogLikelihood = sumLogLikelihood - std::log(probInlier + probOutlier);
        }
        return sumLogLikelihood;
    }

    virtual void Terminate(const Data& data, int N, const Model& bestModel) {
        if (dataError2 != NULL) {
            delete[] dataError2;
            dataError2 = NULL;
        }
        RANSAC<Model, Datum, Data>::Terminate(bestModel, data, N);
    }

    int paramIterationEM;

    float paramSigmaScale;

    float* dataError2;

    float dataSigma2;
};

}  // namespace RTL

#endif  // End of '__RTL_MLESAC__'
