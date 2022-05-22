#pragma once

#ifndef ZIMG_GRAPH_FILTERGRAPH_H_
#define ZIMG_GRAPH_FILTERGRAPH_H_

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


typedef int node_id;

typedef std::array<bool, PLANE_NUM> plane_mask;
typedef std::array<node_id, PLANE_NUM> id_map;

constexpr node_id invalid_id = -1;
constexpr id_map null_ids{ { invalid_id, invalid_id, invalid_id, invalid_id } };


class FilterGraph : public zimg_filter_graph {
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
	/**
	 * Construct a blank graph.
	 */
	FilterGraph();

	/**
	 * Move construct a FilterGraph.
	 *
	 * @param other rvalue
	 */
	FilterGraph(FilterGraph &&other) noexcept;

	/**
	 * Destroy graph.
	 */
	~FilterGraph();

	/**
	 * Move assignment
	 *
	 * @param other rvalue
	 * @return this
	 */
	FilterGraph &operator=(FilterGraph &&other) noexcept;

	/**
	 * Add a source node with specified format.
	 *
	 * @param attr image format
	 * @param subsample_w log2 horizontal subsampling
	 * @param subsample_h log2 vertical subsampling
	 * @param planes color components present
	 */
	node_id add_source(const ImageFilter::image_attributes &attr, unsigned subsample_w, unsigned subsample_h, const plane_mask &planes);

	/**
	 * Add a filter node.
	 *
	 * @param filter image filter
	 * @param deps source nodes for each color component
	 * @param output_planes color components produced
	 */
	node_id attach_filter(std::shared_ptr<ImageFilter> filter, const id_map &deps, const plane_mask &output_planes);

	/**
	 * Add a sink node.
	 *
	 * @param deps source nodes for each color component
	 */
	void set_output(const id_map &deps);

	/**
	 * Get size of temporary buffer required to execute graph.
	 *
	 * @return size in bytes
	 */
	size_t get_tmp_size() const;

	/**
	 * Get number of lines required in input buffer.
	 *
	 * @return number of lines
	 */
	unsigned get_input_buffering() const;

	/**
	 * Get number of lines required in output buffer.
	 *
	 * @return number of lines
	 */
	unsigned get_output_buffering() const;

	/**
	 * Get the optimal tile width used for graph execution.
	 *
	 * @return tile width in output pixels
	 */
	unsigned get_tile_width() const;

	/**
	 * Override the tile width used for graph execution.
	 *
	 * @param tile_width tile width in output pixels
	 */
	void set_tile_width(unsigned tile_width);

	/**
	 * Check if the graph requires 64-byte data alignment.
	 *
	 * @return true if 64-byte alignment is required, else false
	 */
	bool requires_64b_alignment() const;

	/**
	 * Set the graph as requiring 64-byte data alignment.
	 */
	void set_requires_64b_alignment();

	/**
	 * Process an image frame with filter graph.
	 *
	 * @param src pointer to input buffers
	 * @param dst pointer to output buffers
	 * @param tmp temporary buffer
	 * @param unpack_cb user-defined input callback
	 * @param pack_cb user-defined output callback
	 */
	void process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_FILTERGRAPH_H_
