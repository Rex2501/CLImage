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
 NOTE: Some of this code is derived from OpenCV
 */

#include "homography.hpp"

#include "SVD.hpp"

namespace gls {

template <typename T>
gls::image<T> MatrixMultiply(const gls::image<T>& X1, const gls::image<T>& X2) {
    assert(X1.width == X2.height);

    auto result = gls::image<T>(X2.width, X1.height);

    for (int i = 0; i < result.height; i++) {
        for (int j = 0; j < result.width; j++) {
            T sum = 0;
            for (int ki = 0; ki < X1.width; ki++) {
                sum += X1[i][ki] * X2[ki][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

template <typename T>
gls::image<T> transpose(const gls::image<T>& a) {
    auto at = gls::image<T>(a.height, a.width);
    for (int j = 0; j < a.height; j++) {
        for (int i = 0; i < a.width; i++) {  // find the transpose of a
            at[i][j] = a[j][i];
        }
    }
    return at;
}

template <typename T, size_t N>
gls::Vector<N, T> minEigenvectorSVD(gls::Matrix<N, N, T> A) {
    gls::Matrix<N, N, T> tU;
    gls::Vector<N, T> tS;
    gls::Matrix<N, N, T> tVT;

    // Compute SVD
    svd(A, tS, tU, tVT);

    // Find the index of the smallest eigenvalue
    double min_s = std::numeric_limits<double>::max();
    int idx_s = 0;
    for (int i = 0; i < N; i++) {
        if (tS[i] < min_s) {
            min_s = tS[i];
            idx_s = i;
        }
    }

    return tU[idx_s];
}

template <typename TP>
void buildHomogeneousLinearLeastSquaresMatrix(const std::vector<Point2f>& src, const std::vector<Point2f>& dst, TP& P) {
    const int count = (int) src.size();
    assert(dst.size() == count);
    assert(P.width == 9 && P.height == 2 * count);

    for (int i = 0; i < count; i++) {
        const gls::basic_point<double> p1 = { src[i].x, src[i].y };
        const gls::basic_point<double> p2 = { dst[i].x, dst[i].y };

        gls::Vector<9, double> dx = { -p1.x,   -p1.y, -1, 0, 0, 0, p1.x * p2.x, p1.y * p2.x, p2.x };
        gls::Vector<9, double> dy = { 0, 0, 0, -p1.x, -p1.y, -1,   p1.x * p2.y, p1.y * p2.y, p2.y };

        for (int j = 0; j < 9; j++) {
            P[2 * i][j]     = dx[j];
            P[2 * i + 1][j] = dy[j];
        }
    }
}

gls::Matrix<3, 3> getPerspectiveTransformSVD(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    gls::Matrix<9, 9, double> ata;

    if (src.size() == 4 && dst.size() == 4) {
        gls::Matrix<8, 9, double> A;
        buildHomogeneousLinearLeastSquaresMatrix(src, dst, A);

        ata = transpose(A) * A;
    } else {
        int count = (int) src.size();

        auto A = gls::image<double>(9, 2 * count);
        buildHomogeneousLinearLeastSquaresMatrix(src, dst, A);

        /* Normal matrix to find least squares */
        const auto iat = transpose(A);
        const auto iata = MatrixMultiply(iat, A);

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                ata[i][j] = iata[i][j];
            }
        }
    }

    // Convert solution eigenvector to a 3x3 Matrix
    const auto H = gls::Matrix<3, 3, float>(minEigenvectorSVD(ata));
    return H / H[2][2];
}

template <typename TA, typename TB>
void getPerspectiveTransformAB(const std::vector<Point2f>& src, const std::vector<Point2f>& dst, TA& a, TB& b) {
    const int count = (int) src.size();
    assert(a.width == 8 && a.height == 2 * count);
    assert(b.width == 1 && b.height == 2 * count);

    for (int i = 0; i < count; i++) {
        const auto& p1 = src[i];
        const auto& p2 = dst[i];

        a[i][0] = a[i + count][3] = p1.x;
        a[i][1] = a[i + count][4] = p1.y;
        a[i][2] = a[i + count][5] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i + count][0] = a[i + count][1] = a[i + count][2] = 0;
        a[i][6] = -p1.x * p2.x;
        a[i][7] = -p1.y * p2.x;
        a[i + count][6] = -p1.x * p2.y;
        a[i + count][7] = -p1.y * p2.y;
        b[i][0] = p2.x;
        b[i + count][0] = p2.y;
    }
}

gls::Matrix<3, 3> getPerspectiveTransformLSM2(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    int count = (int) src.size();
    gls::Matrix<8, 8> ata;
    gls::Vector<8> atb;

    if (count == 4) {
        gls::Matrix<8, 8> a;
        gls::Matrix<8, 1> b;
        getPerspectiveTransformAB(src, dst, a, b);

        const auto at = transpose(a);
        ata = at * a;
        atb = gls::Vector<8> (at * b);
    } else {
        auto a = gls::image<float>(8, 2 * count);
        auto b = gls::image<float>(1, 2 * count);

        getPerspectiveTransformAB(src, dst, a, b);

        /* Normal matrix to find least squares */
        const auto iat = transpose(a);
        const auto iata = MatrixMultiply(iat, a);
        const auto iatb = MatrixMultiply(iat, b);

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                ata[i][j] = iata[i][j];
            }
        }
        for (int i = 0; i < 8; i++) {
            atb[i] = iatb[i][0];
        }
    }

    // return inverse(aa) * bb;
    const auto h = LUSolve(ata, atb);

    return {
        h[0], h[1], h[2],
        h[3], h[4], h[5],
        h[6], h[7], 1
    };
}

// Jacobi Solver - Computes Eigenvalues and Eigenvectors of Symmetric Square Matrices

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
            // Convergence test
            if (std::abs(p) <= eps) {
                break;
            }

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
    const int count = (int) M.size();

    typedef gls::Vector<2, double> vec2;

    auto cM = vec2::zeros();
    auto cm = vec2::zeros();
    auto sM = vec2::zeros();
    auto sm = vec2::zeros();

    // Find the barycenter of the point set
    for (int i = 0; i < count; i++) {
        cm += vec2(m[i]);
        cM += vec2(M[i]);
    }
    cm /= (double) count;
    cM /= (double) count;

    // Find the mean distance from the barycenter
    for (int i = 0; i < count; i++) {
        sm += abs(vec2(m[i]) - cm);
        sM += abs(vec2(M[i]) - cM);
    }
    sm /= (double) count;
    sM /= (double) count;

    const auto eps = std::numeric_limits<double>::epsilon();
    if (fabs(sm[0]) < eps || fabs(sm[1]) < eps || fabs(sM[0]) < eps || fabs(sM[1]) < eps) {
        throw std::range_error("Can't generate homography from input points.");
    }

    // Efficiently build the Homogeneous Linear Least Squares Matrix (LtL = transpose(L) * L)
    auto LtL = gls::Matrix<9, 9, double>::zeros();
    for (int i = 0; i < count; i++) {
        // Scale and Normalize the point set coordinates
        const auto p1 = (vec2(M[i]) - cM) / sM;
        const auto p2 = (vec2(m[i]) - cm) / sm;

        const gls::Vector<9, double> Lx = { p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] };
        const gls::Vector<9, double> Ly = { 0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] };

        // Only build the top diagonal elements, replicate the rest with completeSymm (see below)
        for (int j = 0; j < 9; j++) {
            for (int k = j; k < 9; k++) {
                LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
            }
        }
    }
    completeSymm(LtL);

#if USE_SVD
    const auto H0 = gls::Matrix<3, 3, double>(minEigenvectorSVD(LtL));
#else
    gls::Vector<9, double> matW;
    gls::Matrix<9, 9, double> matV;
    Jacobi(LtL, &matW, &matV);
    // Convert solution eigenvector to a 3x3 Matrix
    const auto H0 = gls::Matrix<3, 3, double>(matV[8]);
#endif

    const gls::Matrix<3, 3, double> invHnorm = {
        sm[0], 0, cm[0],
        0, sm[1], cm[1],
        0, 0, 1
    };
    const gls::Matrix<3, 3, double> Hnorm2 = {
        1 / sM[0], 0, -cM[0] / sM[0],
        0, 1 / sM[1], -cM[1] / sM[1],
        0, 0, 1
    };

    // Invert the point set coordinates scaling
    const auto H = invHnorm * H0 * Hnorm2;

    // Convert to a float 3x3 Matrix and normalize
    return gls::Matrix<3, 3, float>(H) / (float) H[2][2];
}

}  // namespace gls
