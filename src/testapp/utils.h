#pragma once

#ifndef UTILS_H_
#define UTILS_H_

#include <memory>
#include <utility>
#include <vector>

namespace graphengine {
class Filter;
}


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

class FilterExecutor_GE {
public:
	static constexpr int ALL_PLANES = -1;
	static constexpr int CHROMA_PLANES = -2;
private:
	struct data;

	std::unique_ptr<data> m_data;
public:
	FilterExecutor_GE(const graphengine::Filter *filter, const ImageFrame *src_frame, ImageFrame *dst_frame);

	FilterExecutor_GE(const std::vector<std::pair<int, const graphengine::Filter *>> &filters, const ImageFrame *src_frame, ImageFrame *dst_frame);

	FilterExecutor_GE(FilterExecutor_GE &&other) noexcept;

	~FilterExecutor_GE();

	FilterExecutor_GE &operator=(FilterExecutor_GE &&other) noexcept;

	void operator()();
};

#endif // UTILS_H_
