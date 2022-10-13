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

#ifndef SVD_h
#define SVD_h

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "gls_linalg.hpp"

template <size_t N, size_t M, class T>
void GivensL(gls::Matrix<N, M, T>& S_, size_t m, T a, T b) {
    T r = std::sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

// #pragma omp parallel for
    for (size_t i = 0; i < M; i++) {
        T S0 = S_[m + 0][i];
        T S1 = S_[m + 1][i];
        S_[m][i] += S0 * (c - 1);
        S_[m][i] += S1 * (-s);

        S_[m + 1][i] += S0 * (s);
        S_[m + 1][i] += S1 * (c - 1);
    }
}

template <size_t N, size_t M, class T>
void GivensR(gls::Matrix<N, M, T>& S_, size_t m, T a, T b) {
    T r = std::sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

// #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        T S0 = S_[i][m + 0];
        T S1 = S_[i][m + 1];
        S_[i][m] += S0 * (c - 1);
        S_[i][m] += S1 * (-s);

        S_[i][m + 1] += S0 * (s);
        S_[i][m + 1] += S1 * (c - 1);
    }
}

template <size_t N, size_t M, class T>
void SVD(gls::Matrix<N, N, T>& U_, gls::Matrix<N, M, T>& S_, gls::Matrix<M, M, T>& V_, T eps = -1) {
    static_assert(N >= M);

    {  // Bi-diagonalization
        size_t n = std::min(N, M);
        std::vector<T> house_vec(std::max(N, M));
        for (size_t i = 0; i < n; i++) {
            // Column Householder
            {
                T x1 = S_[i][i];
                if (x1 < 0) x1 = -x1;

                T x_inv_norm = 0;
                for (size_t j = i; j < N; j++) {
                    x_inv_norm += S_[j][i] * S_[j][i];
                }
                if (x_inv_norm > 0) x_inv_norm = 1 / std::sqrt(x_inv_norm);

                T alpha = std::sqrt(1 + x1 * x_inv_norm);
                T beta = x_inv_norm / alpha;

                house_vec[i] = -alpha;
                for (size_t j = i + 1; j < N; j++) {
                    house_vec[j] = -beta * S_[j][i];
                }
                if (S_[i][i] < 0)
                    for (size_t j = i + 1; j < N; j++) {
                        house_vec[j] = -house_vec[j];
                    }
            }
// #pragma omp parallel for
            for (size_t k = i; k < M; k++) {
                T dot_prod = 0;
                for (size_t j = i; j < N; j++) {
                    dot_prod += S_[j][k] * house_vec[j];
                }
                for (size_t j = i; j < N; j++) {
                    S_[j][k] -= dot_prod * house_vec[j];
                }
            }
// #pragma omp parallel for
            for (size_t k = 0; k < N; k++) {
                T dot_prod = 0;
                for (size_t j = i; j < N; j++) {
                    dot_prod += U_[k][j] * house_vec[j];
                }
                for (size_t j = i; j < N; j++) {
                    U_[k][j] -= dot_prod * house_vec[j];
                }
            }

            // Row Householder
            if (i >= n - 1) continue;
            {
                T x1 = S_[i][i + 1];
                if (x1 < 0) x1 = -x1;

                T x_inv_norm = 0;
                for (size_t j = i + 1; j < M; j++) {
                    x_inv_norm += S_[i][j] * S_[i][j];
                }
                if (x_inv_norm > 0) x_inv_norm = 1 / std::sqrt(x_inv_norm);

                T alpha = std::sqrt(1 + x1 * x_inv_norm);
                T beta = x_inv_norm / alpha;

                house_vec[i + 1] = -alpha;
                for (size_t j = i + 2; j < M; j++) {
                    house_vec[j] = -beta * S_[i][j];
                }
                if (S_[i][i + 1] < 0)
                    for (size_t j = i + 2; j < M; j++) {
                        house_vec[j] = -house_vec[j];
                    }
            }
// #pragma omp parallel for
            for (size_t k = i; k < N; k++) {
                T dot_prod = 0;
                for (size_t j = i + 1; j < M; j++) {
                    dot_prod += S_[k][j] * house_vec[j];
                }
                for (size_t j = i + 1; j < M; j++) {
                    S_[k][j] -= dot_prod * house_vec[j];
                }
            }
// #pragma omp parallel for
            for (size_t k = 0; k < M; k++) {
                T dot_prod = 0;
                for (size_t j = i + 1; j < M; j++) {
                    dot_prod += V_[j][k] * house_vec[j];
                }
                for (size_t j = i + 1; j < M; j++) {
                    V_[j][k] -= dot_prod * house_vec[j];
                }
            }
        }
    }

    size_t k0 = 0;
    if (eps < 0) {
        eps = 1.0;
        while (eps + (T)1.0 > 1.0) eps *= 0.5;
        eps *= 64.0;
    }
    while (k0 < M - 1) {  // Diagonalization
        T S_max = 0.0;
        for (size_t i = 0; i < M; i++) S_max = (S_max > S_[i][i] ? S_max : S_[i][i]);

        while (k0 < M - 1 && fabs(S_[k0][k0 + 1]) <= eps * S_max) k0++;
        if (k0 == M - 1) continue;

        size_t n = k0 + 2;
        while (n < M && fabs(S_[n - 1][n]) > eps * S_max) n++;

        T alpha = 0;
        T beta = 0;
        {  // Compute mu
            T C[2][2];
            C[0][0] = S_[n - 2][n - 2] * S_[n - 2][n - 2];
            if (n - k0 > 2) C[0][0] += S_[n - 3][n - 2] * S_[n - 3][n - 2];
            C[0][1] = S_[n - 2][n - 2] * S_[n - 2][n - 1];
            C[1][0] = S_[n - 2][n - 2] * S_[n - 2][n - 1];
            C[1][1] = S_[n - 1][n - 1] * S_[n - 1][n - 1] + S_[n - 2][n - 1] * S_[n - 2][n - 1];

            T b = -(C[0][0] + C[1][1]) / 2;
            T c = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            T d = 0;
            if (b * b - c > 0)
                d = std::sqrt(b * b - c);
            else {
                T b = (C[0][0] - C[1][1]) / 2;
                T c = -C[0][1] * C[1][0];
                if (b * b - c > 0) d = std::sqrt(b * b - c);
            }

            T lambda1 = -b + d;
            T lambda2 = -b - d;

            T d1 = lambda1 - C[1][1];
            d1 = (d1 < 0 ? -d1 : d1);
            T d2 = lambda2 - C[1][1];
            d2 = (d2 < 0 ? -d2 : d2);
            T mu = (d1 < d2 ? lambda1 : lambda2);

            alpha = S_[k0][k0] * S_[k0][k0] - mu;
            beta = S_[k0][k0] * S_[k0][k0 + 1];
        }

        for (size_t k = k0; k < n - 1; k++) {
            GivensR(S_, k, alpha, beta);
            GivensL(V_, k, alpha, beta);

            alpha = S_[k][k];
            beta = S_[k + 1][k];
            GivensL(S_, k, alpha, beta);
            GivensR(U_, k, alpha, beta);

            alpha = S_[k][k + 1];
            beta = S_[k][k + 2];
        }

        {  // Make S bi-diagonal again
            for (size_t i0 = k0; i0 < n - 1; i0++) {
                for (size_t i1 = 0; i1 < M; i1++) {
                    if (i0 > i1 || i0 + 1 < i1) S_[i0][i1] = 0;
                }
            }
            for (size_t i0 = 0; i0 < N; i0++) {
                for (size_t i1 = k0; i1 < n - 1; i1++) {
                    if (i0 > i1 || i0 + 1 < i1) S_[i0][i1] = 0;
                }
            }
            for (size_t i = 0; i < M - 1; i++) {
                if (fabs(S_[i][i + 1]) <= eps * S_max) {
                    S_[i][i + 1] = 0;
                }
            }
        }
    }
}

template <size_t N, size_t M, size_t DMIN = std::min(M, N), class T>
inline void svd(const gls::Matrix<N, M, T>& A, gls::Vector<DMIN, T>& S, gls::Matrix<DMIN, M, T>& U, gls::Matrix<N, DMIN, T>& VT) {
    const constexpr size_t DMAX = std::max(M, N);

    auto U_ = gls::Matrix<DMAX, DMAX, T>::zeros();
    auto V_ = gls::Matrix<DMIN, DMIN, T>::zeros();
    auto S_ = gls::Matrix<DMAX, DMIN, T>();

    if (DMIN == M) {
        for (size_t i = 0; i < DMAX; i++)
            for (size_t j = 0; j < DMIN; j++) {
                S_[i][j] = A[i][j];
            }
    } else {
        for (size_t i = 0; i < DMAX; i++)
            for (size_t j = 0; j < DMIN; j++) {
                S_[i][j] = A[j][i];
            }
    }
    for (size_t i = 0; i < DMAX; i++) {
        U_[i][i] = 1;
    }
    for (size_t i = 0; i < DMIN; i++) {
        V_[i][i] = 1;
    }

    SVD<DMAX, DMIN, T>(U_, S_, V_, (T)-1);

    for (size_t i = 0; i < DMIN; i++) {  // Set S
        S[i] = S_[i][i];
    }
    if (DMIN == M) {  // Set U
        for (size_t i = 0; i < DMIN; i++)
            for (size_t j = 0; j < M; j++) {
                U[i][j] = V_[i][j] * (S[i] < 0.0 ? -1.0 : 1.0);
            }
    } else {
        for (size_t i = 0; i < DMIN; i++)
            for (size_t j = 0; j < M; j++) {
                U[i][j] = U_[j][i] * (S[i] < 0.0 ? -1.0 : 1.0);
            }
    }
    if (DMAX == N) {  // Set V
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < DMIN; j++) {
                VT[i][j] = U_[i][j];
            }
    } else {
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < DMIN; j++) {
                VT[i][j] = V_[j][i];
            }
    }
    for (size_t i = 0; i < DMIN; i++) {
        S[i] = S[i] * (S[i] < 0.0 ? -1.0 : 1.0);
    }
}

#endif /* SVD_h */
