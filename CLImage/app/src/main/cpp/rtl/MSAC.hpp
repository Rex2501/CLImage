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

#ifndef __RTL_MSAC__
#define __RTL_MSAC__

#include "RANSAC.hpp"

namespace RTL {

template <class Model, class Datum, class Data>
class MSAC : virtual public RANSAC<Model, Datum, Data> {
   public:
    MSAC(Estimator<Model, Datum, Data>* estimator) : RANSAC<Model, Datum, Data>(estimator) {}

   protected:
    inline float EvaluateModel(const Model& model, const Data& data, int N) override {
        float loss = 0;
        for (int i = 0; i < N; i++) {
            float error = RANSAC<Model, Datum, Data>::toolEstimator->ComputeError(model, data[i]);
            const auto paramThreshold = this->paramThreshold;
            if (std::abs(error) > paramThreshold)
                loss += paramThreshold * paramThreshold;
            else
                loss += error * error;
        }
        return loss;
    }
};

}  // namespace RTL

#endif  // End of '__RTL_MSAC__'
