#pragma once

#ifndef ZIMG_UNIT_TEST_GRAPH_AUDIT_BUFFER_H_
#define ZIMG_UNIT_TEST_GRAPH_AUDIT_BUFFER_H_

#include <cstdint>
#include "common/alloc.h"
#include "common/pixel.h"
#include "graph/image_buffer.h"

template <class T>
class AuditBuffer {
	zimg::AlignedVector<T> m_vector[3];
	zimg::graph::ImageBuffer<T> m_buffer[3];
	zimg::PixelFormat m_format;
	unsigned m_width[3];
	unsigned m_buffer_height[3];
	unsigned m_subsample_w;
	unsigned m_subsample_h;
	T m_fill_val[3];
	T m_guard_val;
	bool m_color;

	static T splat_byte(unsigned char b);

	void add_guard_bytes();

	ptrdiff_t stride_T(unsigned p) const;
public:
	AuditBuffer(unsigned width, unsigned height, const zimg::PixelFormat &format, unsigned lines,
	            unsigned subsample_w, unsigned subsample_h, bool color);

	void set_fill_val(unsigned char x);

	void set_fill_val(unsigned char x, unsigned plane);

	bool detect_write(unsigned i, unsigned left, unsigned right) const;

	void assert_eq(const AuditBuffer &other, unsigned i, unsigned left, unsigned right) const;

	void assert_guard_bytes() const;

	void random_fill(unsigned first_row, unsigned last_row, unsigned first_col, unsigned last_col);

	void default_fill();

	zimg::graph::ColorImageBuffer<const void> as_read_buffer() const;

	zimg::graph::ColorImageBuffer<void> as_write_buffer() const;
};

extern template class AuditBuffer<uint8_t>;
extern template class AuditBuffer<uint16_t>;
extern template class AuditBuffer<float>;

#endif // ZIMG_UNIT_TEST_GRAPH_AUDIT_BUFFER_H_
