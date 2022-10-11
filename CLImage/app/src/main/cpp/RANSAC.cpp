//
//  RANSAC.cpp
//  ImageRegistration
//
//  Created by Fabio Riccardi on 9/21/22.
//

#include "RANSAC.hpp"

#include <float.h>

#include "gls_linalg.hpp"

#define PSEUDO_RANDOM_TEST_SEQUENCE true

#if PSEUDO_RANDOM_TEST_SEQUENCE
#include "PRNG.h"
#endif

namespace gls {

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
    int count = (int) src.size();

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

template <size_t NN=8>
gls::Vector<NN> getPerspectiveTransformIata(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    gls::Matrix<NN, NN> a;
    gls::Matrix<NN, 1> b;
    getPerspectiveTransformAB(src, dst, a, b);

    gls::Matrix<NN, NN> at = transpose(a);
    gls::Matrix<NN, NN> ata = at * a;
    gls::Vector<NN> atb = at * b;

    // TODO: replace this with a proper solver
    // return inverse(ata) * atb;
    return LUSolve(ata, atb);
}

template <size_t NN=8>
gls::Vector<NN> getPerspectiveTransformLSM2(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    int count = (int) src.size();

    auto a = gls::image<float>(NN, 2 * count);
    auto b = gls::image<float>(1, 2 * count);

    getPerspectiveTransformAB(src, dst, a, b);

    auto at = gls::image<float>(a.height, NN);
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < 2 * count; j++) {
            at[i][j] = a[j][i];
        }
    }

    const auto ata = MatrixMultiply(at, a);
    const auto atb = MatrixMultiply(at, b);

    gls::Matrix<NN, NN> aa;
    gls::Vector<NN> bb;
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            aa[i][j] = ata[i][j];
        }
    }
    for (int i = 0; i < NN; i++) {
        bb[i] = atb[i][0];
    }

    // TODO: replace this with a proper solver
    // return inverse(aa) * bb;
    return LUSolve(aa, bb);
}

gls::Vector<8> getRANSAC2(const std::vector<Point2f>& p1, const std::vector<Point2f>& p2, float threshold, int count) {
    assert(p1.size() > 0 && p2.size() > 0);

    // Calculate the maximum set of interior points
    int max_iters = 2000;
    int iters = max_iters;
    int innerP, max_innerP = 0;
    std::vector<int> innerPvInd;  // Inner point set index - temporary
    std::vector<int> innerPvInd_i;
    std::vector<Point2f> selectP1, selectP2;
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
        selectP1.push_back(p1[selectIndex[k][0]]);
        selectP1.push_back(p1[selectIndex[k][1]]);
        selectP1.push_back(p1[selectIndex[k][2]]);
        selectP1.push_back(p1[selectIndex[k][3]]);
        selectP2.push_back(p2[selectIndex[k][0]]);
        selectP2.push_back(p2[selectIndex[k][1]]);
        selectP2.push_back(p2[selectIndex[k][2]]);
        selectP2.push_back(p2[selectIndex[k][3]]);

        // Calculate the perspective transformation matrix
        // (currently the singular matrix inverse cannot be solved, and the
        // singular decomposition inversion method will be improved in the future)
        try {
            gls::Vector<8> trans = getPerspectiveTransformIata(selectP1, selectP2);

            // Calculate the model parameter error, if the error is greater than the threshold, discard
            // this set of model parameters
            innerP = 0;
            float u, v, w;
            float errX, errY;
            for (int i = 0; i < p1.size(); i++) {
                errX = errY = 0;
                u = p1[i].x * trans[0] + p1[i].y * trans[1] + trans[2];
                v = p1[i].x * trans[3] + p1[i].y * trans[4] + trans[5];
                w = p1[i].x * trans[6] + p1[i].y * trans[7] + 1;
                errX = fabs(u / w - p2[i].x);
                errY = fabs(v / w - p2[i].y);
                if (threshold > (errX * errX + errY * errY)) {
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

        selectP1.clear();
        selectP2.clear();
    }
    printf(" RANSAC interior point ratio - number of loops: %d %ld %d \t\n ", max_innerP, p1.size(), k);

    // Calculate projection matrix parameters based on interior points

    std::vector<Point2f> _p1(max_innerP), _p2(max_innerP);
    for (int i = 0; i < max_innerP; i++) {
        _p1[i] = p1[innerPvInd_i[i]];
        _p2[i] = p2[innerPvInd_i[i]];
    }

    return getPerspectiveTransformLSM2(_p1, _p2);
}

} // namespace gls
