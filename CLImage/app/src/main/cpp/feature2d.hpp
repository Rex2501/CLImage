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
