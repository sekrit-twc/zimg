#pragma once

#ifndef UTILS_H_
#define UTILS_H_

#include <memory>

namespace zimg {
namespace graph {

class ImageFilter;

} // namespace graph
} // namespace zimg


class ImageFrame;

class FilterExecutor {
	struct data;

	std::shared_ptr<data> m_data;
	const zimg::graph::ImageFilter *m_filter;
	const zimg::graph::ImageFilter *m_filter_uv;
	const ImageFrame *m_src_frame;
	ImageFrame *m_dst_frame;

	void exec_grey(const zimg::graph::ImageFilter *filter, unsigned plane);
	void exec_color();
public:
	FilterExecutor(const zimg::graph::ImageFilter *filter, const zimg::graph::ImageFilter *filter_uv, const ImageFrame *src_frame, ImageFrame *dst_frame);

	void operator()();
};

#endif // UTILS_H_
