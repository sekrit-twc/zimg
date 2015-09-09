#pragma once

#ifndef ZIMG_FILTERGRAPH_H
#define ZIMG_FILTERGRAPH_H

#include <memory>
#include "ztypes.h"

namespace zimg {;

class IZimgFilter;

enum class PixelType;

class FilterGraph {
	class impl;
public:
	class callback {
	public:
		typedef int (*func_type)(void *user, unsigned i, unsigned left, unsigned right);
	private:
		func_type m_func;
		void *m_user;
	public:
		callback(std::nullptr_t x = nullptr);

		callback(func_type func, void *user);

		explicit operator bool() const;

		void operator()(unsigned i, unsigned left, unsigned right) const;

		friend class FilterGraph;
	};
private:
	std::unique_ptr<impl> m_impl;
public:
	FilterGraph(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color);

	~FilterGraph();

	void attach_filter(IZimgFilter *filter);

	void attach_filter_uv(IZimgFilter *filter);

	void complete();

	size_t get_tmp_size() const;

	unsigned get_input_buffering() const;

	unsigned get_output_buffering() const;

	void process(const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *tmp, callback unpack_cb, callback pack_cb) const;
};

} // namespace zimg

#endif // ZIMG_FILTERGRAPH_H
