//
//  SURF.hpp
//  RawPipeline
//
//  Created by Fabio Riccardi on 8/12/22.
//

#ifndef SURF_hpp
#define SURF_hpp

#include "gls_image.hpp"
#include "gls_linalg.hpp"

namespace surf {

typedef gls::basic_point<float> Point2f;

bool mSURF_Detection(const gls::image<float>& srcIMAGE1, const gls::image<float>& srcIMAGE2,
                     std::vector<Point2f>* matchpoints1, std::vector<Point2f>* matchpoints2, int matches_num);

std::vector<float> getRANSAC2(const std::vector<Point2f>& p1, const std::vector<Point2f>& p2, float threshold, int count);

template <size_t N, typename baseT>
gls::Matrix<N, N, baseT> inverse(const gls::Matrix<N, N, baseT>& m);

} // namespace surf

#endif /* SURF_hpp */
