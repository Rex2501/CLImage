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

#include "gls_cl_image.hpp"
#include "gls_linalg.hpp"

namespace gls {

typedef gls::basic_point<float> Point2f;

bool SURF_Detection(gls::OpenCLContext* cLContext, const gls::image<float>& srcIMAGE1, const gls::image<float>& srcIMAGE2,
                    std::vector<Point2f>* matchpoints1, std::vector<Point2f>* matchpoints2);

void clRegisterAndFuse(gls::OpenCLContext* cLContext,
                       const gls::cl_image_2d<gls::rgba_pixel>& inputImage0,
                       const gls::cl_image_2d<gls::rgba_pixel>& inputImage1,
                       gls::cl_image_2d<gls::rgba_pixel>* outputImage,
                       const gls::Matrix<3, 3>& homography);

} // namespace surf

#endif /* SURF_hpp */
