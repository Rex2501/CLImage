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

#ifndef RANSAC_hpp
#define RANSAC_hpp

#include <vector>

#include "feature2d.hpp"
#include "gls_linalg.hpp"

namespace gls {

gls::Matrix<3, 3> RANSAC(const std::vector<std::pair<Point2f, Point2f>> matchpoints, float threshold, int max_iterations, std::vector<int>* inlier_indices = nullptr);

} // namespace gls

#endif /* RANSAC_hpp */
