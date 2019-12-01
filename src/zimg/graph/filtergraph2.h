#pragma once

#ifndef ZIMG_GRAPH_FILTERGRAPH2_H_
#define ZIMG_GRAPH_FILTERGRAPH2_H_

#include <array>
#include <memory>
#include "image_filter.h"

// Base class in global namespace for API export.
struct zimg_filter_graph {
	virtual inline ~zimg_filter_graph() = 0;
};

zimg_filter_graph::~zimg_filter_graph() = default;


namespace zimg {

enum class PixelType;

namespace graph {

class ImageFilter;

template <class T>
class ImageBuffer;

constexpr int PLANE_NUM = 4;
constexpr int PLANE_Y = 0;
constexpr int PLANE_U = 1;
constexpr int PLANE_V = 2;
constexpr int PLANE_A = 3;

class FilterGraph2 : public zimg_filter_graph {
	class impl;
public:
	/**
	 * User-defined I/O callback functor.
	 */
	class callback {
		typedef int (*func_type)(void *user, unsigned i, unsigned left, unsigned right);

		func_type m_func;
		void *m_user;
	public:
		/**
		 * Default construct callback, creating a null callback.
		 */
		callback(std::nullptr_t x = nullptr);

		/**
		 * Construct a callback from user-defined function.
		 *
		 * @param func function pointer
		 * @param user user private data
		 */
		callback(func_type func, void *user);

		/**
		 * Check if callback is set.
		 *
		 * @return true if callback is not null, else false
		 */
		explicit operator bool() const;

		/**
		 * Invoke user-defined callback.
		 *
		 * @param i row index of line to read/write
		 * @param left left column index
		 * @param right right column index, plus one
		 */
		void operator()(unsigned i, unsigned left, unsigned right) const;
	};
private:
	std::unique_ptr<impl> m_impl;

	impl *get_impl() noexcept { return m_impl.get(); }
	const impl *get_impl() const noexcept { return m_impl.get(); }
public:
	typedef int node_id;

	typedef ImageFilter::image_attributes image_attributes;
	typedef std::array<bool, PLANE_NUM> plane_mask;
	typedef std::array<node_id, PLANE_NUM> id_map;

	FilterGraph2();

	FilterGraph2(FilterGraph2 &&other) noexcept;

	~FilterGraph2();

	FilterGraph2 &operator=(FilterGraph2 &&other) noexcept;

	node_id add_source(const image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes);

	node_id attach_filter(std::shared_ptr<ImageFilter> filter, const id_map &deps, const plane_mask &output_planes);

	void set_output(const id_map &deps);

	size_t get_tmp_size() const;

	unsigned get_input_buffering() const;

	unsigned get_output_buffering() const;

	void process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_FILTERGRAPH2_H_
