#pragma once

#ifndef UTILS_H_
#define UTILS_H_

#include <memory>

namespace zimg {;

class IZimgFilter;

} // namespace zimg


class ImageFrame;

class FilterExecutor {
	struct data;

	std::shared_ptr<data> m_data;
	const zimg::IZimgFilter *m_filter;
	const zimg::IZimgFilter *m_filter_uv;
	const ImageFrame *m_src_frame;
	ImageFrame *m_dst_frame;

	void exec_grey(const zimg::IZimgFilter *filter, unsigned plane);
	void exec_color();
public:
	FilterExecutor(const zimg::IZimgFilter *filter, const zimg::IZimgFilter *filter_uv, const ImageFrame *src_frame, ImageFrame *dst_frame);

	void operator()();
};

#endif // UTILS_H_
