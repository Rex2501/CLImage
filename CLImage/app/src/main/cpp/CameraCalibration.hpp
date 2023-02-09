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

#ifndef CameraCalibration_hpp
#define CameraCalibration_hpp

#include <filesystem>

#include "demosaic.hpp"
#include "raw_converter.hpp"

template <size_t levels = 5>
class CameraCalibration {
public:
    virtual ~CameraCalibration() {}

    virtual NoiseModel<levels> nlfFromIso(int iso) const = 0;

    virtual std::pair<float, std::array<DenoiseParameters, levels>> getDenoiseParameters(int iso) const = 0;

    virtual void calibrate(RawConverter* rawConverter, const std::filesystem::path& input_dir) const = 0;

    virtual DemosaicParameters buildDemosaicParameters() const = 0;

    gls::image<gls::rgb_pixel>::unique_ptr calibrate(RawConverter* rawConverter,
                                                     const std::filesystem::path& input_path,
                                                     DemosaicParameters* demosaicParameters,
                                                     int iso, const gls::rectangle& gmb_position) const {
        gls::tiff_metadata dng_metadata, exif_metadata;
        const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

        unpackDNGMetadata(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false, /* &gmb_position */ nullptr, /*rotate_180=*/ false);

        // See if the ISO value is present and override
        if (getValue(exif_metadata, EXIFTAG_RECOMMENDEDEXPOSUREINDEX, (uint32_t*) &iso)) {
            iso = iso;
        } else {
            iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];
        }

        const auto denoiseParameters = getDenoiseParameters(iso);
        demosaicParameters->noiseLevel = denoiseParameters.first;
        demosaicParameters->denoiseParameters = denoiseParameters.second;

        return RawConverter::convertToRGBImage(*rawConverter->runPipeline(*inputImage, demosaicParameters, /*calibrateFromImage=*/ true));
    }

    std::unique_ptr<DemosaicParameters> getDemosaicParameters(const gls::image<gls::luma_pixel_16>& inputImage,
                                                              gls::tiff_metadata* dng_metadata,
                                                              gls::tiff_metadata* exif_metadata) const {
        auto demosaicParameters = std::make_unique<DemosaicParameters>();

        *demosaicParameters = buildDemosaicParameters();

        unpackDNGMetadata(inputImage, dng_metadata, demosaicParameters.get(), /*auto_white_balance=*/ false, /* &gmb_position */ nullptr, /*rotate_180=*/ false);

        uint32_t iso = 0;
        std::vector<uint16_t> iso_16;
        if (!(iso_16 = getVector<uint16_t>(*dng_metadata, TIFFTAG_ISO)).empty()) {
            iso = iso_16[0];
        } else if (!(iso_16 = getVector<uint16_t>(*exif_metadata, EXIFTAG_ISOSPEEDRATINGS)).empty()) {
            iso = iso_16[0];
        } else if (getValue(*exif_metadata, EXIFTAG_RECOMMENDEDEXPOSUREINDEX, &iso)) {
            iso = iso;
        }

        std::cout << "EXIF ISO: " << iso << std::endl;

        const auto nlfParams = nlfFromIso(iso);
        const auto denoiseParameters = getDenoiseParameters(iso);
        demosaicParameters->noiseModel = nlfParams;
        demosaicParameters->noiseLevel = denoiseParameters.first;
        demosaicParameters->denoiseParameters = denoiseParameters.second;

        return demosaicParameters;
    }
};

std::unique_ptr<CameraCalibration<5>> getIPhone11Calibration();

std::unique_ptr<CameraCalibration<5>> getLeicaQ2Calibration();

gls::image<gls::rgb_pixel>::unique_ptr demosaicIMX571DNG(RawConverter* rawConverter, const std::filesystem::path& input_path);
void calibrateIMX571(RawConverter* rawConverter, const std::filesystem::path& input_dir);

gls::image<gls::rgb_pixel>::unique_ptr demosaicLeicaQ2DNG(RawConverter* rawConverter, const std::filesystem::path& input_path);
void calibrateLeicaQ2(RawConverter* rawConverter, const std::filesystem::path& input_dir);

gls::image<gls::rgb_pixel>::unique_ptr demosaicCanonEOSRPDNG(RawConverter* rawConverter, const std::filesystem::path& input_path);
void calibrateCanonEOSRP(RawConverter* rawConverter, const std::filesystem::path& input_dir);

gls::image<gls::rgb_pixel>::unique_ptr demosaicSonya6400DNG(RawConverter* rawConverter, const std::filesystem::path& input_path);
void calibrateSonya6400(RawConverter* rawConverter, const std::filesystem::path& input_dir);

template <typename T = gls::rgb_pixel>
typename gls::image<T>::unique_ptr demosaicSonya6400RawImage(RawConverter* rawConverter,
                                                             gls::tiff_metadata* dng_metadata,
                                                             gls::tiff_metadata* exif_metadata,
                                                             const gls::image<gls::luma_pixel_16>& inputImage);

void calibrateRicohGRIII(RawConverter* rawConverter, const std::filesystem::path& input_dir);
gls::image<gls::rgb_pixel>::unique_ptr demosaicRicohGRIII2DNG(RawConverter* rawConverter, const std::filesystem::path& input_path);

void calibrateiPhone11(RawConverter* rawConverter, const std::filesystem::path& input_dir);
gls::image<gls::rgb_pixel>::unique_ptr demosaiciPhone11(RawConverter* rawConverter, const std::filesystem::path& input_path);

#endif /* CameraCalibration_hpp */
