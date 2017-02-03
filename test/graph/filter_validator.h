#pragma once

#ifndef ZIMG_UNIT_TEST_GRAPH_FILTER_VALIDATOR_H_
#define ZIMG_UNIT_TEST_GRAPH_FILTER_VALIDATOR_H_

#include "common/pixel.h"

namespace zimg {
namespace graph {

class ImageFilter;

} // namespace graph
} // namespace zimg


class FilterValidator {
	const zimg::graph::ImageFilter *m_test_filter;
	const zimg::graph::ImageFilter *m_ref_filter;

	zimg::PixelFormat m_src_format;
	unsigned m_src_width;
	unsigned m_src_height;

	const char * const *m_sha1_str;
	double m_snr_thresh;
public:
	FilterValidator(const zimg::graph::ImageFilter *test_filter, unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format);

	FilterValidator &set_ref_filter(const zimg::graph::ImageFilter *ref_filter, double snr_thresh);
	FilterValidator &set_sha1(const char * const sha1_str[3]);

	void validate();
};


bool assert_different_dynamic_type(const zimg::graph::ImageFilter *filter_a, const zimg::graph::ImageFilter *filter_b);

#endif // ZIMG_UNIT_TEST_GRAPH_FILTER_VALIDATOR_H_
