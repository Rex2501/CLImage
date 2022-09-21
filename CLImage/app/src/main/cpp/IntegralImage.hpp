//
//  IntegralImage.hpp
//  ImageRegistration
//
//  Created by Fabio Riccardi on 9/21/22.
//

#ifndef IntegralImage_hpp
#define IntegralImage_hpp

#include "gls_image.hpp"

#include "gls_cl_image.hpp"

namespace gls {

template <typename T>
void integral(const gls::image<float>& img, gls::image<T>* sum, const float bias = 1) {
    // Zero the first row and the first column of the sum
    for (int i = 0; i < sum->width; i++) {
        (*sum)[0][i] = 0;
    }
    for (int j = 1; j < sum->height; j++) {
        (*sum)[j][0] = 0;
    }

    for (int j = 1; j < sum->height; j++) {
        for (int i = 1; i < sum->width; i++) {
            (*sum)[j][i] = img[j - 1][i - 1] / bias + (*sum)[j][i - 1] + (*sum)[j - 1][i] - (*sum)[j - 1][i - 1];
        }
    }
}

void clIntegral(gls::OpenCLContext* glsContext, const gls::image<float>& img, const gls::cl_image_buffer_2d<float>* sum, const float bias = 1);

}

#endif /* IntegralImage_hpp */
