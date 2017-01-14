#pragma once

#ifndef TABLE_H_
#define TABLE_H_

#include <memory>
#include "common/static_map.h"

namespace zimg {

enum class CPUClass;
enum class PixelType;

namespace colorspace {

enum class MatrixCoefficients;
enum class TransferCharacteristics;
enum class ColorPrimaries;

} // namespace colorspace

namespace depth {

enum class DitherType;

} // namespace depth

namespace resize {

class Filter;

} // namespace resize

} // namespace zimg


extern const zimg::static_string_map<zimg::CPUClass, 7> g_cpu_table;
extern const zimg::static_string_map<zimg::PixelType, 4> g_pixel_table;
extern const zimg::static_string_map<zimg::colorspace::MatrixCoefficients, 7> g_matrix_table;
extern const zimg::static_string_map<zimg::colorspace::TransferCharacteristics, 6> g_transfer_table;
extern const zimg::static_string_map<zimg::colorspace::ColorPrimaries, 5> g_primaries_table;
extern const zimg::static_string_map<zimg::depth::DitherType, 4> g_dither_table;
extern const zimg::static_string_map<std::unique_ptr<zimg::resize::Filter>(*)(double, double), 7> g_resize_table;

#endif // TABLE_H_
