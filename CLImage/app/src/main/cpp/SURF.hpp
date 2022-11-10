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

#ifndef SURF_hpp
#define SURF_hpp

#include <float.h>

#include "gls_cl_image.hpp"
#include "gls_linalg.hpp"

#include "feature2d.hpp"

namespace gls {

typedef gls::basic_point<float> Point2f;

class DMatch {
   public:
    DMatch() : queryIdx(-1), trainIdx(-1), distance(FLT_MAX) {}
    DMatch(int _queryIdx, int _trainIdx, float _distance)
        : queryIdx(_queryIdx), trainIdx(_trainIdx), distance(_distance) {}

    int queryIdx;  // query descriptor index
    int trainIdx;  // train descriptor index

    float distance;

    // less is better
    bool operator < (const DMatch& m) const { return distance < m.distance; }
};

class SURF {
public:
    static std::unique_ptr<SURF> makeInstance(gls::OpenCLContext* glsContext, int width, int height,
                                              int max_features = -1, int nOctaves = 4,
                                              int nOctaveLayers = 2, float hessianThreshold = 0.02);

    virtual ~SURF() {}

    virtual void integral(const gls::image<float>& img, const std::array<gls::cl_image_2d<float>::unique_ptr, 4>& sum) = 0;

    virtual void detect(const std::array<gls::cl_image_2d<float>::unique_ptr, 4>& integralSum, std::vector<KeyPoint>* keypoints) = 0;

    virtual void detectAndCompute(const gls::image<float>& img, std::vector<KeyPoint>* keypoints, gls::image<float>::unique_ptr* _descriptors) = 0;

    virtual std::vector<DMatch> matchKeyPoints(const gls::image<float>& descriptor1, const gls::image<float>& descriptor2) = 0;

    static std::vector<std::pair<Point2f, Point2f>> detection(gls::OpenCLContext* cLContext, const gls::image<float>& image1, const gls::image<float>& image2);
};

void clRegisterAndFuse(gls::OpenCLContext* cLContext,
                       const gls::cl_image_2d<gls::rgba_pixel>& inputImage0,
                       const gls::cl_image_2d<gls::rgba_pixel>& inputImage1,
                       gls::cl_image_2d<gls::rgba_pixel>* outputImage,
                       const gls::Matrix<3, 3>& homography);

template <typename T>
void clRegisterImage(gls::OpenCLContext* cLContext,
                     const gls::cl_image_2d<T>& inputImage,
                     gls::cl_image_2d<T>* outputImage,
                     const gls::Matrix<3, 3>& homography);

} // namespace surf

#endif /* SURF_hpp */
