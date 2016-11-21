#pragma once

#ifndef ZIMG_GRAPH_FILTERGRAPH_H_
#define ZIMG_GRAPH_FILTERGRAPH_H_

#include <memory>

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


/**
 * Manages dynamic traversal and execution of filter graphs.
 */
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
	 * Construct a FilterGraph from a specific source format.
	 *
	 * @param width source width
	 * @param height source height
	 * @param type source pixel type
	 * @param subsample_w log2 horizontal subsampling
	 * @param subsample_h log2 vertical subsampling
	 * @param color if source is a color image
	 */
	FilterGraph(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color);

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
	 * Move assignment.
	 *
	 * @param other rvalue
	 * @return this
	 */
	FilterGraph &operator=(FilterGraph &&other) noexcept;

	/**
	 * Attach a filter to the graph, transferring ownership.
	 *
	 * If the filter is a color filter, it is applied to all image channels, else
	 * it is applied to the luma channel only.
	 *
	 * @param filter image filter
	 */
	void attach_filter(std::unique_ptr<ImageFilter> &&filter);

	/**
	 * Attach a filter to the graph chroma channels.
	 *
	 * @see attach_filter
	 */
	void attach_filter_uv(std::unique_ptr<ImageFilter> &&filter);

	/**
	 * Discard the chroma channels of the graph.
	 */
	void color_to_grey();

	/**
	 * Convert the graph from grey to color.
	 *
	 * If {@p yuv} is true, then the chroma channels are initialized to the zero
	 * value of the specified pixel format, else the grey channel is replicated.
	 *
	 * @param yuv whether to create a YUV or RGB image
	 * @param subsample_w log2 horizontal subsampling
	 * @param subsample_h log2 vertical subsampling
	 * @param depth source bit depth
	 */
	void grey_to_color(bool yuv, unsigned subsample_w, unsigned subsample_h, unsigned depth);

	/**
	 * Finalize graph.
	 *
	 * No additional filters may be attached to a finalized graph. Upon calling
	 * this method, the dynamic execution strategy will be computed.
	 */
	void complete();

	/**
	 * Get size of temporary buffer required to execute graph.
	 *
	 * @return size in bytes
	 */
	size_t get_tmp_size() const;

	/**
	 * Get number of input lines used simultaneously during graph execution.
	 *
	 * @return number of lines
	 */
	unsigned get_input_buffering() const;

	/**
	 * Get number of output lines used simultaneously during graph execution.
	 *
	 * @return number of lines
	 */
	unsigned get_output_buffering() const;

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
