#pragma once

#ifndef ZIMG_MOCK_FILTER_H_
#define ZIMG_MOCK_FILTER_H_

#include <cstdint>

#include "Common/zfilter.h"

class MockFilter : public zimg::IZimgFilter {
protected:
	struct context {
		unsigned last_line;
		unsigned last_left;
		unsigned last_right;
	};

	image_attributes m_attr;
	zimg::ZimgFilterFlags m_flags;
	mutable unsigned m_total_calls;
	unsigned m_simultaneous_lines;
	unsigned m_horizontal_support;
	unsigned m_vertical_support;
public:
	MockFilter(unsigned width, unsigned height, zimg::PixelType type, const zimg::ZimgFilterFlags &flags = {});

	unsigned get_total_calls() const;

	void set_simultaneous_lines(unsigned n);

	void set_horizontal_support(unsigned n);

	void set_vertical_support(unsigned n);

	// IZimgFilter
	zimg::ZimgFilterFlags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_simultaneous_lines() const override;

	unsigned get_max_buffering() const override;

	size_t get_context_size() const override;

	size_t get_tmp_size(unsigned left, unsigned right) const override;

	void init_context(void *ctx) const override;

	void process(void *ctx, const zimg::ZimgImageBufferConst &src, const zimg::ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

template <class T>
class SplatFilter : public MockFilter {
	static T splat_byte(unsigned char b);

	T m_src_val;
	T m_dst_val;
	bool m_input_checking;
public:
	SplatFilter(unsigned width, unsigned height, zimg::PixelType type, const zimg::ZimgFilterFlags &flags = {});

	void set_input_val(unsigned char x);

	void set_output_val(unsigned char x);

	void enable_input_checking(bool enabled);

	void process(void *ctx, const zimg::ZimgImageBufferConst &src, const zimg::ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

extern template class SplatFilter<uint8_t>;
extern template class SplatFilter<uint16_t>;
extern template class SplatFilter<float>;

#endif // ZIMG_MOCK_FILTER_H_
