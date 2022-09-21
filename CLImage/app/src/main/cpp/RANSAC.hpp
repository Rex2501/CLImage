//
//  RANSAC.hpp
//  ImageRegistration
//
//  Created by Fabio Riccardi on 9/21/22.
//

#ifndef RANSAC_hpp
#define RANSAC_hpp

#include <vector>

#include "feature2d.hpp"

namespace gls {

std::vector<float> getRANSAC2(const std::vector<Point2f>& p1, const std::vector<Point2f>& p2, float threshold, int count);

} // namespace gls

#endif /* RANSAC_hpp */
