#ifndef RESIZE_H_
#define RESIZE_H_

#include <memory>
#include "filter.h"
#include "osdep.h"

namespace resize {

class ResizeImpl;

class Resize {
	int m_src_width;
	int m_src_height;
	int m_dst_width;
	int m_dst_height;
	bool m_skip_h;
	bool m_skip_v;
	std::shared_ptr<ResizeImpl> m_impl;
public:
	Resize() = default;

	Resize(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
	       double shift_w, double shift_h, double subwidth, double subheight, bool x86);

	~Resize();

	size_t tmp_size() const;

	void process(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, int src_stride, int dst_stride) const;
};

} // namespace resize

#endif // RESIZE_H_
