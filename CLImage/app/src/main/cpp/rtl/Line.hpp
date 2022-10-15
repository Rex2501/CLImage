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

#ifndef __RTL_LINE__
#define __RTL_LINE__

#include <cmath>
#include <limits>
#include <random>

#include "Base.hpp"

class Point {
   public:
    Point() : x(0), y(0) {}

    Point(float _x, float _y) : x(_x), y(_y) {}

    friend std::ostream& operator<<(std::ostream& out, const Point& p) {
        return out << p.x << ", " << p.y;
    }

    float x, y;
};

class Line {
   public:
    Line() : a(0), b(0), c(0) {}

    Line(float _a, float _b, float _c) : a(_a), b(_b), c(_c) {}

    friend std::ostream& operator<<(std::ostream& out, const Line& l) {
        return out << l.a << ", " << l.b << ", " << l.c;
    }

    float a, b, c;
};

class LineEstimator : virtual public RTL::Estimator<Line, Point, std::vector<Point> > {
   public:
    virtual Line ComputeModel(const std::vector<Point>& data, const std::set<int>& samples) {
        float meanX = 0, meanY = 0, meanXX = 0, meanYY = 0, meanXY = 0;
        for (auto itr = samples.begin(); itr != samples.end(); itr++) {
            const Point& p = data[*itr];
            meanX += p.x;
            meanY += p.y;
            meanXX += p.x * p.x;
            meanYY += p.y * p.y;
            meanXY += p.x * p.y;
        }
        size_t M = samples.size();
        meanX /= M;
        meanY /= M;
        meanXX /= M;
        meanYY /= M;
        meanXY /= M;
        float a = meanXX - meanX * meanX;
        float b = meanXY - meanX * meanY;
        float d = meanYY - meanY * meanY;

        Line line;
        if (std::abs(b) > FLT_EPSILON) {
            // Calculate the first eigen vector of A = [a, b; b, d]
            // Ref. http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
            float T2 = (a + d) / 2;
            float lambda = T2 - std::sqrt(T2 * T2 - (a * d - b * b));
            float v1 = lambda - d, v2 = b;
            float norm = std::sqrt(v1 * v1 + v2 * v2);
            line.a = v1 / norm;
            line.b = v2 / norm;
        } else {
            line.a = 1;
            line.b = 0;
        }
        line.c = -line.a * meanX - line.b * meanY;
        return line;
    }

    virtual float ComputeError(const Line& line, const Point& point) {
        return line.a * point.x + line.b * point.y + line.c;
    }
};  // End of 'LineEstimator'

class LineObserver : virtual public RTL::Observer<Line, Point, std::vector<Point> > {
   public:
    LineObserver(Point _max = Point(640, 480), Point _min = Point(0, 0))
        : RANGE_MIN(_min), RANGE_MAX(_max) {}

    virtual std::vector<Point> GenerateData(const Line& line, int N, std::vector<int>& inliers,
                                            float noise = 0, float ratio = 1) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> uniform(0, 1);
        std::normal_distribution<float> normal(0, 1);

        std::vector<Point> data;
        if (std::abs(line.b) > std::abs(line.a)) {
            for (int i = 0; i < N; i++) {
                Point point;
                point.x = (RANGE_MAX.x - RANGE_MIN.x) * uniform(generator) + RANGE_MIN.x;
                float vote = uniform(generator);
                if (vote > ratio) {
                    // Generate an outlier
                    point.y = (RANGE_MAX.y - RANGE_MIN.y) * uniform(generator) + RANGE_MIN.y;
                } else {
                    // Generate an inlier
                    point.y = (line.a * point.x + line.c) / -line.b;
                    point.x += noise * normal(generator);
                    point.y += noise * normal(generator);
                    inliers.push_back(i);
                }
                data.push_back(point);
            }
        } else {
            for (int i = 0; i < N; i++) {
                Point point;
                point.y = (RANGE_MAX.y - RANGE_MIN.y) * uniform(generator) + RANGE_MIN.y;
                float vote = uniform(generator);
                if (vote > ratio) {
                    // Generate an outlier
                    point.x = (RANGE_MAX.x - RANGE_MIN.x) * uniform(generator) + RANGE_MIN.x;
                } else {
                    // Generate an inlier
                    point.x = (line.b * point.y + line.c) / -line.a;
                    point.x += noise * normal(generator);
                    point.y += noise * normal(generator);
                    inliers.push_back(i);
                }
                data.push_back(point);
            }
        }
        return data;
    }

    const Point RANGE_MIN;

    const Point RANGE_MAX;
};

#endif  // End of '__RTL_LINE__'
