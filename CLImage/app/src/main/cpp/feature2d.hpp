//
//  feature2d.hpp.h
//  RawPipeline
//
//  Created by Fabio Riccardi on 8/19/22.
//

#ifndef feature2d_hpp
#define feature2d_hpp

#include <cmath>

#include "gls_image.hpp"

typedef gls::basic_point<float> Point2f;

struct KeyPoint {
    Point2f pt;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;

    KeyPoint()
        : pt(0,0), size(0), angle(-1), response(0), octave(0), class_id(-1) {}

    KeyPoint(Point2f _pt, float _size, float _angle, float _response, int _octave, int _class_id)
        : pt(_pt), size(_size), angle(_angle), response(_response), octave(_octave), class_id(_class_id) {}

    KeyPoint(float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)
        : pt(x, y), size(_size), angle(_angle), response(_response), octave(_octave), class_id(_class_id) {}

    bool operator == (const KeyPoint& other) const {
        return pt == other.pt && size == other.size &&
               angle == other.angle && response == other.response &&
               octave == other.octave && class_id == other.class_id;
    }
};

inline int cvRound(float x) {
    return (int) lrintf(x);
}

#endif /* feature2d_hpp */
