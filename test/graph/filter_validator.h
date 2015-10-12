#pragma once

#ifndef ZIMG_UNIT_TEST_GRAPH_FILTER_VALIDATOR_H_
#define ZIMG_UNIT_TEST_GRAPH_FILTER_VALIDATOR_H_

namespace zimg {;

struct PixelFormat;

namespace graph {;

class ImageFilter;

} // namespace graph
} // namespace zimg


void validate_filter(const zimg::graph::ImageFilter *filter, unsigned src_width, unsigned src_height,
                     const zimg::PixelFormat &src_format, const char * const sha1_str[3] = nullptr);

void validate_filter_reference(const zimg::graph::ImageFilter *ref_filter, const zimg::graph::ImageFilter *test_filter,
                               unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, double snr_thresh);

#endif // ZIMG_UNIT_TEST_GRAPH_FILTER_VALIDATOR_H_
