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

#ifndef GLS_IMAGE_H
#define GLS_IMAGE_H

#include <sys/types.h>

#include <cassert>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "gls_geometry.hpp"
#include "gls_image_jpeg.h"
#include "gls_image_png.h"
#include "gls_image_tiff.h"

#define USE_FP16_FLOATS true

namespace gls {

template <typename T>
struct basic_luma_pixel {
    constexpr static int channels = 1;
    constexpr static int bit_depth = 8 * sizeof(T);
    typedef T dataType;

    union {
        struct {
            T luma;
        };
        struct {
            T x;
        };
        std::array<T, channels> v;
    };

    basic_luma_pixel() {}
    basic_luma_pixel(T _luma) : luma(_luma) {}
    basic_luma_pixel(T _v[channels]) : v(_v) {}
    basic_luma_pixel(const std::array<T, channels>& _v) { std::copy(_v.begin(), _v.end(), v.begin()); }

    operator T() const { return luma; }

    T& operator[](int c) { return v[c]; }
    const T& operator[](int c) const { return v[c]; }
};

template <typename T>
struct basic_luma_alpha_pixel {
    constexpr static int channels = 2;
    constexpr static int bit_depth = 8 * sizeof(T);
    typedef T dataType;

    union {
        struct {
            T luma;
            T alpha;
        };
        struct {
            T x;
            T y;
        };
        std::array<T, channels> v;
    };

    basic_luma_alpha_pixel() {}
    basic_luma_alpha_pixel(T _luma, T _alpha) : luma(_luma), alpha(_alpha) {}
    basic_luma_alpha_pixel(T _v[channels]) : v(_v) {}
    basic_luma_alpha_pixel(const std::array<T, channels>& _v) { std::copy(_v.begin(), _v.end(), v.begin()); }

    T& operator[](int c) { return v[c]; }
    const T& operator[](int c) const { return v[c]; }
};

template <typename T>
struct basic_rgb_pixel {
    constexpr static size_t channels = 3;
    constexpr static int bit_depth = 8 * sizeof(T);
    typedef T dataType;

    union {
        struct {
            T red;
            T green;
            T blue;
        };
        struct {
            T x;
            T y;
            T z;
        };
        std::array<T, channels> v;
    };

    basic_rgb_pixel() {}
    basic_rgb_pixel(T _red, T _green, T _blue) : red(_red), green(_green), blue(_blue) {}
    basic_rgb_pixel(const T _v[channels]) : v(_v) {}
    basic_rgb_pixel(const std::array<T, channels>& _v) { std::copy(_v.begin(), _v.end(), v.begin()); }

    T& operator[](int c) { return v[c]; }
    const T& operator[](int c) const { return v[c]; }
};

template <typename T>
struct basic_rgba_pixel {
    constexpr static int channels = 4;
    constexpr static int bit_depth = 8 * sizeof(T);
    typedef T dataType;

    union {
        struct {
            T red;
            T green;
            T blue;
            T alpha;
        };
        struct {
            T x;
            T y;
            T z;
            T w;
        };
        std::array<T, channels> v;
    };

    basic_rgba_pixel() {}
    basic_rgba_pixel(T _red, T _green, T _blue, T _alpha) : red(_red), green(_green), blue(_blue), alpha(_alpha) {}
    basic_rgba_pixel(T _v[channels]) : v(_v) {}
    basic_rgba_pixel(const std::array<T, channels>& _v) { std::copy(_v.begin(), _v.end(), v.begin()); }

    T& operator[](int c) { return v[c]; }
    const T& operator[](int c) const { return v[c]; }
};

typedef basic_luma_pixel<uint8_t> luma_pixel;
typedef basic_luma_alpha_pixel<uint8_t> luma_alpha_pixel;
typedef basic_rgb_pixel<uint8_t> rgb_pixel;
typedef basic_rgba_pixel<uint8_t> rgba_pixel;

typedef basic_luma_pixel<uint16_t> luma_pixel_16;
typedef basic_luma_alpha_pixel<uint16_t> luma_alpha_pixel_16;
typedef basic_rgb_pixel<uint16_t> rgb_pixel_16;
typedef basic_rgba_pixel<uint16_t> rgba_pixel_16;

typedef basic_luma_pixel<float> luma_pixel_fp32;
typedef basic_luma_alpha_pixel<float> luma_alpha_pixel_fp32;
typedef basic_rgb_pixel<float> rgb_pixel_fp32;
typedef basic_rgba_pixel<float> rgba_pixel_fp32;

typedef basic_luma_pixel<float> pixel_fp32;
typedef basic_luma_alpha_pixel<float> pixel_fp32_2;
typedef basic_rgb_pixel<float> pixel_fp32_3;
typedef basic_rgba_pixel<float> pixel_fp32_4;

#if USE_FP16_FLOATS && !(__APPLE__ && TARGET_CPU_X86_64)
typedef __fp16 float16_t;
typedef basic_luma_pixel<float16_t> luma_pixel_fp16;
typedef basic_luma_alpha_pixel<float16_t> luma_alpha_pixel_fp16;
typedef basic_rgb_pixel<float16_t> rgb_pixel_fp16;
typedef basic_rgba_pixel<float16_t> rgba_pixel_fp16;
#endif

#if USE_FP16_FLOATS && !(__APPLE__ && TARGET_CPU_X86_64)
typedef float16_t float_type;
#else
typedef float float_type;
#endif
typedef basic_luma_pixel<float_type> luma_pixel_float;
typedef basic_luma_alpha_pixel<float_type> luma_alpha_pixel_float;
typedef basic_rgb_pixel<float_type> rgb_pixel_float;
typedef basic_rgba_pixel<float_type> rgba_pixel_float;

typedef basic_luma_pixel<float_type> pixel_float;
typedef basic_luma_alpha_pixel<float_type> pixel_float2;
typedef basic_rgb_pixel<float_type> pixel_float3;
typedef basic_rgba_pixel<float_type> pixel_float4;

class tiff_metadata;

template <typename T>
class basic_image {
   public:
    const int width;
    const int height;

    gls::size size() const {
        return { width, height };
    }

    static const constexpr int pixel_bit_depth = T::bit_depth;
    static const constexpr int pixel_channels = T::channels;
    static const constexpr int pixel_size = sizeof(T);

    typedef std::unique_ptr<basic_image<T>> unique_ptr;

    basic_image(int _width, int _height) : width(_width), height(_height) {}
    basic_image(gls::size _dimensions) : width(_dimensions.width), height(_dimensions.height) {}
};

// std::vector is convenient but it is expensive as it initializes memory
#define USE_STD_VECTOR_ALLOCATION false

template <typename T>
class image : public basic_image<T> {
   public:
    const int stride;
    typedef std::unique_ptr<image<T>> unique_ptr;

   protected:
#if USE_STD_VECTOR_ALLOCATION
    const std::unique_ptr<std::vector<T>> _data_store = nullptr;
#else
    T* _data_store = nullptr;
#endif
    const std::span<T> _data;

   public:
    // Data is owned by the image and retained by _data_store
    image(int _width, int _height, int _stride)
        : basic_image<T>(_width, _height),
          stride(_stride),
#if USE_STD_VECTOR_ALLOCATION
          _data_store(std::make_unique<std::vector<T>>(_stride * _height)),
          _data(_data_store->data(), _data_store->size()) {}
#else
          _data_store(new T[_stride * _height]),
          _data(_data_store, _stride * _height) {}

    virtual ~image() {
        delete _data_store;
    }
#endif

    image(int _width, int _height) : image(_width, _height, _width) {}

    image(size _dimensions) : image(_dimensions.width, _dimensions.height) {}

    // Data is owned by caller, the image is only a wrapper around it
    image(int _width, int _height, int _stride, std::span<T> data)
        : basic_image<T>(_width, _height), stride(_stride), _data(data) {
        assert(_stride * _height <= data.size());
    }

    image(int _width, int _height, std::span<T> data) : image<T>(_width, _height, _width, data) {}

    image(image *_base, int _x, int _y, int _width, int _height) : image<T>(_width, _height, _base->stride, std::span(_base->_data.data() + _y * _base->stride + _x, _base->stride * _height)) {
        assert(_x + _width <= _base->width && _y + _height <= _base->height);
    }

    image(image *_base, const rectangle& _crop) : image(_base, _crop.x, _crop.y, _crop.width, _crop.height) {}

    image(const image& _base, int _x, int _y, int _width, int _height) : image<T>(_width, _height, _base.stride, std::span(_base._data.data() + _y * _base.stride + _x, _base.stride * _height)) {
        assert(_x + _width <= _base.width && _y + _height <= _base.height);
    }

    image(const image& _base, const rectangle& _crop) : image(_base, _crop.x, _crop.y, _crop.width, _crop.height) {}

    // row access
    T* operator[](int row) { return &_data[stride * row]; }

    const T* operator[](int row) const { return &_data[stride * row]; }

    const std::span<T> pixels() const { return _data; }

    void apply(std::function<void(const T& pixel)> process) const {
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process((*this)[y][x]);
            }
        }
    }

    void apply(std::function<void(const T& pixel, int x, int y)> process) const {
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process((*this)[y][x], x, y);
            }
        }
    }

    void apply(std::function<void(T* pixel, int x, int y)> process) {
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process(&(*this)[y][x], x, y);
            }
        }
    }

    const size_t size_in_bytes() { return _data.size() * basic_image<T>::pixel_size; }

    // image factory from PNG file
    static unique_ptr read_png_file(const std::string& filename) {
        unique_ptr image = nullptr;

        auto image_allocator = [&image](int width, int height, std::vector<uint8_t*>* row_pointers) -> bool {
            if ((image = std::make_unique<gls::image<T>>(width, height)) == nullptr) {
                return false;
            }
            for (int i = 0; i < height; ++i) {
                (*row_pointers)[i] = (uint8_t*)(*image)[i];
            }
            return true;
        };

        gls::read_png_file(filename, T::channels, T::bit_depth, image_allocator);

        return image;
    }

    // Write image to PNG file
    // compression_level range: [0-9], 0 -> no compression (default), 1 -> *fast* compression, otherwise useful range: [3-6]
    void write_png_file(const std::string& filename, int compression_level = 0) const {
        auto row_pointer = [this](int row) -> uint8_t* { return (uint8_t*)(*this)[row]; };
        gls::write_png_file(filename, basic_image<T>::width, basic_image<T>::height, T::channels, T::bit_depth,
                            false, compression_level, row_pointer);
    }

    void write_png_file(const std::string& filename, bool skip_alpha, int compression_level = 0) const {
        auto row_pointer = [this](int row) -> uint8_t* { return (uint8_t*)(*this)[row]; };
        gls::write_png_file(filename, basic_image<T>::width, basic_image<T>::height, T::channels, T::bit_depth,
                            skip_alpha, compression_level, row_pointer);
    }

    // Image factory from JPEG file
    static unique_ptr read_jpeg_file(const std::string& filename) {
        unique_ptr image = nullptr;

        auto image_allocator = [&image](int width, int height) -> std::span<uint8_t> {
            if ((image = std::make_unique<gls::image<T>>(width, height)) == nullptr) {
                return std::span<uint8_t>();
            }
            return std::span<uint8_t>((uint8_t*)(*image)[0], sizeof(T) * width * height);
        };

        gls::read_jpeg_file(filename, T::channels, T::bit_depth, image_allocator);

        return image;
    }

    // Write image to JPEG file
    void write_jpeg_file(const std::string& filename, int quality) const {
        auto image_data = [this]() -> std::span<uint8_t> {
            return std::span<uint8_t>((uint8_t*)this->_data.data(), sizeof(T) * this->_data.size());
        };
        gls::write_jpeg_file(filename, basic_image<T>::width, basic_image<T>::height, stride, T::channels, T::bit_depth,
                             image_data, quality);
    }

    // Helper function for read_tiff_file and read_dng_file
    static bool process_tiff_strip(image* destination, int tiff_bitspersample, int tiff_samplesperpixel,
                                   int destination_row, int strip_width, int strip_height,
                                   int crop_x, int crop_y, uint8_t *tiff_buffer) {
        typedef typename T::dataType dataType;

        std::function<dataType()> nextTiffPixelSame = [&tiff_buffer]() -> dataType {
            dataType pixelValue = *((dataType *) tiff_buffer);
            tiff_buffer += sizeof(dataType);
            return pixelValue;
        };
        std::function<dataType()> nextTiffPixel8to16 = [&tiff_buffer]() -> dataType {
            return (dataType) *(tiff_buffer++) << 8;;
        };
        std::function<dataType()> nextTiffPixel16to8 = [&tiff_buffer]() -> dataType {
            dataType pixelValue = (dataType) (*((uint16_t *) tiff_buffer) >> 8);
            tiff_buffer += sizeof(uint16_t);
            return pixelValue;
        };

        auto nextTiffPixel = tiff_bitspersample == T::bit_depth
            ? nextTiffPixelSame
            : (tiff_bitspersample == 8)
                ? nextTiffPixel8to16
                : nextTiffPixel16to8;

        for (int y = 0; y < strip_height && y + destination_row - crop_y < destination->height; y++) {
            for (int x = 0; x < strip_width; x++) {
                for (int c = 0; c < std::min((int) tiff_samplesperpixel, (int) T::channels); c++) {
                    if (x >= crop_x && y + destination_row >= crop_y && x - crop_x < destination->width) {
                        (*destination)[y + destination_row - crop_y][x - crop_x][c] = nextTiffPixel();
                    } else {
                        nextTiffPixel();
                    }
                }
            }
        }
        return true;
    };

    // Image factory from TIFF file
    static unique_ptr read_tiff_file(const std::string& filename, tiff_metadata* metadata = nullptr) {
        unique_ptr image = nullptr;
        gls::read_tiff_file(filename, T::channels, T::bit_depth, metadata,
                            [&image](int width, int height) -> bool {
                                return (image = std::make_unique<gls::image<T>>(width, height)) != nullptr;
                            },
                            [&image](int tiff_bitspersample, int tiff_samplesperpixel,
                                     int row, int strip_width, int strip_height,
                                     int crop_x, int crop_y, uint8_t *tiff_buffer) -> bool {
                                return process_tiff_strip(image.get(), tiff_bitspersample, tiff_samplesperpixel,
                                                          row, /*strip_width=*/ image->width, strip_height,
                                                          /*crop_x=*/ 0, /*crop_y=*/ 0, tiff_buffer);
                            });
        return image;
    }

    // Write image to TIFF file
    void write_tiff_file(const std::string& filename, tiff_compression compression = tiff_compression::NONE, tiff_metadata* metadata = nullptr) const {
        typedef typename T::dataType dataType;
        auto row_pointer = [this](int row) -> dataType* { return (dataType*)(*this)[row]; };
        gls::write_tiff_file<dataType>(filename, basic_image<T>::width, basic_image<T>::height, T::channels, T::bit_depth,
                                       compression, metadata, row_pointer);
    }

    // Image factory from DNG file
    static unique_ptr read_dng_file(const std::string& filename, tiff_metadata* dng_metadata = nullptr, tiff_metadata* exif_metadata = nullptr) {
        unique_ptr image = nullptr;
        gls::read_dng_file(filename, T::channels, T::bit_depth, dng_metadata, exif_metadata,
                            [&image](int width, int height) -> bool {
                                return (image = std::make_unique<gls::image<T>>(width, height)) != nullptr;
                            },
                            [&image](int tiff_bitspersample, int tiff_samplesperpixel,
                                     int row, int strip_width, int strip_height,
                                     int crop_x, int crop_y, uint8_t *tiff_buffer) -> bool {
                                return process_tiff_strip(image.get(), tiff_bitspersample, tiff_samplesperpixel,
                                                          row, /*strip_width=*/ strip_width, strip_height,
                                                          /*crop_x=*/ crop_x, /*crop_y=*/ crop_y, tiff_buffer);
                            });
        return image;
    }

    // Write image to DNG file
    void write_dng_file(const std::string& filename, tiff_compression compression = tiff_compression::NONE,
                        const tiff_metadata* dng_metadata = nullptr, const tiff_metadata* exif_metadata = nullptr) const {
        typedef typename T::dataType dataType;
        auto row_pointer = [this](int row) -> dataType* { return (dataType*)(*this)[row]; };
        gls::write_dng_file(filename, basic_image<T>::width, basic_image<T>::height, T::channels, T::bit_depth,
                            compression, dng_metadata, exif_metadata, row_pointer);
    }
};

template <typename T>
static inline void copyPixels(gls::image<T>* to, const gls::image<T>& from) {
    assert(to->width == from.width && to->height == from.height);

    if (to->stride == from.stride) {
        memcpy((void*) (*to)[0], (void*) from[0], to->stride * to->height * sizeof(T));
    } else {
        for (int j = 0; j < to->height; j++) {
            memcpy((void*) (*to)[j], (void*) from[j], to->width * sizeof(T));
        }
    }
}

}  // namespace gls

#endif /* GLS_IMAGE_H */
