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

/*
 NOTE: This code is derived from OpenCV
 */

#include "homography.hpp"

namespace gls {

typedef gls::basic_point<double> Point2d;

// Copies the left or the right half of a square matrix to its another half
template <size_t N, typename T>
void completeSymm(gls::Matrix<N, N, T>& m, bool LtoR = false) {
    int rows = N;
    int j0 = 0, j1 = rows;

    for (int i = 0; i < rows; i++) {
        if (!LtoR) {
            j1 = i;
        } else {
            j0 = i + 1;
        }
        for (int j = j0; j < j1; j++) {
            m[i][j] = m[j][i];
        }
    }
}

template <typename T>
inline void rotate(T* v0, T* v1, T c, T s) {
    T a0 = *v0;
    T b0 = *v1;
    *v0 = a0 * c - b0 * s;
    *v1 = a0 * s + b0 * c;
}

template <size_t N, typename T>
void Jacobi(const gls::Matrix<N, N, T>& A, gls::Vector<N, T>* W, gls::Matrix<N, N, T>* V) {
    auto AW = gls::Matrix<N, N, T>(A);

    const T eps = std::numeric_limits<T>::epsilon();

    if (V) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                (*V)[i][j] = 0;
            }
            (*V)[i][i] = 1;
        }
    }

    const int maxIters = N * N * 30;

    std::array<int, N> indR;
    std::array<int, N> indC;

    for (int k = 0; k < N; k++) {
        (*W)[k] = AW[k][k];
        if (k < N - 1) {
            int m = k + 1;
            T mv = std::abs(AW[k][m]);
            for (int i = k + 2; i < N; i++) {
                T val = std::abs(AW[k][i]);
                if (mv < val) mv = val, m = i;
            }
            indR[k] = m;
        }
        if (k > 0) {
            int m = 0;
            T mv = std::abs(AW[0][k]);
            for (int i = 1; i < k; i++) {
                T val = std::abs(AW[i][k]);
                if (mv < val) mv = val, m = i;
            }
            indC[k] = m;
        }
    }

    if (N > 1) {
        for (int iters = 0; iters < maxIters; iters++) {
            // find index (k,l) of pivot p
            int k = 0;
            T mv = std::abs(AW[0][indR[0]]);
            for (int i = 1; i < N - 1; i++) {
                T val = std::abs(AW[i][indR[i]]);
                if (mv < val) mv = val, k = i;
            }
            int l = indR[k];
            for (int i = 1; i < N; i++) {
                T val = std::abs(AW[indC[i]][i]);
                if (mv < val) mv = val, k = indC[i], l = i;
            }

            T p = AW[k][l];
            if (std::abs(p) <= eps) break;
            T y = (T)(((*W)[l] - (*W)[k]) * 0.5);
            T t = std::abs(y) + hypot(p, y);
            T s = hypot(p, t);
            T c = t / s;
            s = p / s;
            t = (p / t) * p;
            if (y < 0) {
                s = -s;
                t = -t;
            }
            AW[k][l] = 0;

            (*W)[k] -= t;
            (*W)[l] += t;

            // rotate rows and columns k and l
            for (int i = 0; i < k; i++) {
                rotate(&AW[i][k], &AW[i][l], c, s);
            }
            for (int i = k + 1; i < l; i++) {
                rotate(&AW[k][i], &AW[i][l], c, s);
            }
            for (int i = l + 1; i < N; i++) {
                rotate(&AW[k][i], &AW[l][i], c, s);
            }
            // rotate eigenvectors
            if (V) {
                for (int i = 0; i < N; i++) {
                    rotate(&(*V)[k][i], &(*V)[l][i], c, s);
                }
            }

            for (int j = 0; j < 2; j++) {
                int idx = j == 0 ? k : l;
                if (idx < N - 1) {
                    int m = idx + 1;
                    T mv = std::abs(AW[idx][m]);
                    for (int i = idx + 2; i < N; i++) {
                        T val = std::abs(AW[idx][i]);
                        if (mv < val) mv = val, m = i;
                    }
                    indR[idx] = m;
                }
                if (idx > 0) {
                    int m = 0;
                    T mv = std::abs(AW[0][idx]);
                    for (int i = 1; i < idx; i++) {
                        T val = std::abs(AW[i][idx]);
                        if (mv < val) mv = val, m = i;
                    }
                    indC[idx] = m;
                }
            }
        }
    }

    // sort eigenvalues & eigenvectors
    for (int k = 0; k < N - 1; k++) {
        int m = k;
        for (int i = k + 1; i < N; i++) {
            if ((*W)[m] < (*W)[i]) {
                m = i;
            }
        }
        if (k != m) {
            std::swap((*W)[m], (*W)[k]);
            if (V) {
                for (int i = 0; i < N; i++) {
                    std::swap((*V)[m][i], (*V)[k][i]);
                }
            }
        }
    }
}

gls::Matrix<3, 3> findHomography(const std::vector<Point2f>& M, const std::vector<Point2f>& m) {
    const int count = (int)M.size();

    gls::Matrix<9, 9, double> LtL;
    gls::Vector<9, double> matW;
    gls::Matrix<9, 9, double> matV;
    gls::Matrix<3, 3, double> Htemp;

    Point2d cM(0, 0), cm(0, 0), sM(0, 0), sm(0, 0);

    for (int i = 0; i < count; i++) {
        cm.x += m[i].x;
        cm.y += m[i].y;
        cM.x += M[i].x;
        cM.y += M[i].y;
    }

    cm.x /= count;
    cm.y /= count;
    cM.x /= count;
    cM.y /= count;

    for (int i = 0; i < count; i++) {
        sm.x += fabs(m[i].x - cm.x);
        sm.y += fabs(m[i].y - cm.y);
        sM.x += fabs(M[i].x - cM.x);
        sM.y += fabs(M[i].y - cM.y);
    }

    double deps = std::numeric_limits<double>::epsilon();

    if (fabs(sm.x) < deps || fabs(sm.y) < deps || fabs(sM.x) < deps || fabs(sM.y) < deps) {
        throw std::range_error("Solution not available.");
    }

    sm.x = count / sm.x;
    sm.y = count / sm.y;
    sM.x = count / sM.x;
    sM.y = count / sM.y;

    const gls::Matrix<3, 3, double> invHnorm = {1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1};
    const gls::Matrix<3, 3, double> Hnorm2 = {sM.x, 0, -cM.x * sM.x, 0, sM.y, -cM.y * sM.y, 0, 0, 1};

    for (int i = 0; i < count; i++) {
        const double x = (m[i].x - cm.x) * sm.x;
        const double y = (m[i].y - cm.y) * sm.y;
        const double X = (M[i].x - cM.x) * sM.x;
        const double Y = (M[i].y - cM.y) * sM.y;
        const double Lx[] = {X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x};
        const double Ly[] = {0, 0, 0, X, Y, 1, -y * X, -y * Y, -y};

        for (int j = 0; j < 9; j++) {
            for (int k = j; k < 9; k++) {
                LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
            }
        }
    }
    completeSymm(LtL);

    Jacobi(LtL, &matW, &matV);

    gls::Matrix<3, 3, double> H0 = {
        matV[8][0], matV[8][1], matV[8][2],
        matV[8][3], matV[8][4], matV[8][5],
        matV[8][6], matV[8][7], matV[8][8]
    };

    Htemp = invHnorm * H0;
    H0 = Htemp * Hnorm2;

    const auto H0Data = H0.span();
    gls::Matrix<3, 3> model;
    auto modelData = model.span();
    for (int i = 0; i < H0Data.size(); i++) {
        modelData[i] = H0Data[i] / H0[2][2];
    }

    return model;
}

}  // namespace gls
