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

#ifndef gls_geometry_h
#define gls_geometry_h

#include "gls_linalg.hpp"

namespace gls {

template <typename T>
struct basic_point {
    T x;
    T y;
    
    basic_point(T _x, T _y) : x(_x), y(_y) {}
    basic_point() { }
    
    bool operator == (const basic_point& other) const {
        return x == other.x && y == other.y;
    }
    
    basic_point<T> operator + (const T& value) const {
        basic_point<T> result = {
            x + value,
            y + value
        };
        return result;
    }
    
    basic_point<T>& operator += (const T& right) {
        x += right;
        y += right;
        return *this;
    }
    
    basic_point<T> operator - (const T& value) const {
        basic_point<T> result = {
            x - value,
            y - value
        };
        return result;
    }
    
    basic_point<T>& operator -= (const T& right) {
        x -= right;
        y -= right;
        return *this;
    }
    
    basic_point<T> operator * (const T& value) const {
        basic_point<T> result = {
            x * value,
            y * value
        };
        return result;
    }
    
    basic_point<T>& operator *= (const T& right) {
        x *= right;
        y *= right;
        return *this;
    }
    
    basic_point<T> operator / (const T& value) const {
        basic_point<T> result = {
            x / value,
            y / value
        };
        return result;
    }
    
    basic_point<T>& operator /= (const T& right) {
        x /= right;
        y /= right;
        return *this;
    }

    basic_point<T> operator + (const basic_point<T>& other) const {
        basic_point<T> result = {
            x + other.x,
            y + other.y
        };
        return result;
    }

    basic_point<T>& operator += (const basic_point<T>& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    basic_point<T> operator - (const basic_point<T>& other) const {
        basic_point<T> result = {
            x - other.x,
            y - other.y
        };
        return result;
    }

    basic_point<T>& operator -= (const basic_point<T>& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    operator gls::Vector<2, T>() const { return {x, y}; }
};

template <typename value_type>
inline std::ostream& operator<<(std::ostream& os, const basic_point<value_type>& p) {
    os << "x: " << p.x << ", y: " << p.y;
    return os;
}

template <typename T>
bool operator == (const basic_point<T>& a, const basic_point<T>& b) {
    return a.x == b.x && a.y == b.y;
}

template <typename T>
struct basic_size {
    T width;
    T height;
    
    basic_size(T _width, T _height) : width(_width), height(_height) {}
    basic_size() { }
    
    bool operator == (const basic_size& other) const {
        return width == other.width && height == other.height;
    }
    
    basic_size<T> operator + (const T& value) const {
        basic_size<T> result = {
            width + value,
            height + value
        };
        return result;
    }
    
    basic_size<T>& operator += (const T& right) {
        width += right;
        height += right;
        return *this;
    }
    
    basic_size<T> operator - (const T& value) const {
        basic_size<T> result = {
            width - value,
            height - value
        };
        return result;
    }
    
    basic_size<T>& operator -= (const T& right) {
        width -= right;
        height -= right;
        return *this;
    }
    
    basic_size<T> operator * (const T& value) const {
        basic_size<T> result = {
            width * value,
            height * value
        };
        return result;
    }
    
    basic_size<T>& operator *= (const T& right) {
        width *= right;
        height *= right;
        return *this;
    }
    
    basic_size<T> operator / (const T& value) const {
        basic_size<T> result = {
            width / value,
            height / value
        };
        return result;
    }
    
    basic_size<T>& operator /= (const T& right) {
        width /= right;
        height /= right;
        return *this;
    }};

template <typename T>
bool operator == (const basic_size<T>& a, const basic_size<T>& b) {
    return a.width == b.width && a.height == b.height;
}

template <typename T>
struct basic_rectangle : public basic_point<T>, basic_size<T> {
    basic_rectangle(basic_point<T> _origin, basic_size<T> _dimensions) : basic_point<T>(_origin), basic_size<T>(_dimensions) {}
    basic_rectangle(T _x, T _y, T _width, T _height) : basic_point<T>(_x, _y), basic_size<T>(_width, _height) {}
    basic_rectangle() { }
    
    bool contains(const gls::basic_point<T> p) const {
        return p.x >= basic_point<T>::x && p.y >= basic_point<T>::y && p.x < basic_point<T>::x + basic_size<T>::width && p.y < basic_point<T>::y + basic_size<T>::height;
    }
    
    bool operator == (const basic_rectangle& other) const {
        return this->x == other.x && this->y == other.y && this->width == other.width && this->height == other.height;
    }
};

template <typename T>
bool operator == (const basic_rectangle<T>& a, const basic_rectangle<T>& b) {
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

typedef basic_point<int> point;
typedef basic_size<int> size;
typedef basic_rectangle<int> rectangle;

}  // namespace gls

// For maps and sets

template<>
struct std::hash<gls::size> {
    std::size_t operator()(gls::size const& r) const noexcept {
        return r.width ^ r.height;
    }
};


template<>
struct std::hash<gls::point> {
    std::size_t operator()(gls::point const& p) const noexcept {
        return p.x + p.y;
    }
};

#endif /* gls_geometry_h */
