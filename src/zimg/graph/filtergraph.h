#pragma once

#ifndef ZIMG_GRAPH_FILTERGRAPH_H_
#define ZIMG_GRAPH_FILTERGRAPH_H_

#include <memory>
#include "image_buffer.h"

struct zimg_filter_graph {
	virtual inline ~zimg_filter_graph() = 0;
};

zimg_filter_graph::~zimg_filter_graph()
{
}


namespace zimg {;

enum class PixelType;

namespace graph {;

class ImageFilter;


class FilterGraph : public zimg_filter_graph {
	class impl;
public:
	class callback {
		typedef int (*func_type)(void *user, unsigned i, unsigned left, unsigned right);

		func_type m_func;
		void *m_user;
	public:
		callback(std::nullptr_t x = nullptr);

		callback(func_type func, void *user);

		explicit operator bool() const;

		void operator()(unsigned i, unsigned left, unsigned right) const;
	};
private:
	std::unique_ptr<impl> m_impl;
public:
	FilterGraph(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color);

	~FilterGraph();

	void attach_filter(ImageFilter *filter);

	void attach_filter_uv(ImageFilter *filter);

	void color_to_grey();

	void grey_to_color(bool yuv, unsigned subsample_w, unsigned subsample_h, unsigned depth);

	void complete();

	size_t get_tmp_size() const;

	unsigned get_input_buffering() const;

	unsigned get_output_buffering() const;

	void process(const ImageBufferConst &src, const ImageBuffer &dst, void *tmp, callback unpack_cb, callback pack_cb) const;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_FILTERGRAPH_H_
