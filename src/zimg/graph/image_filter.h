#pragma once

#ifndef ZIMG_GRAPH_IMAGE_FILTER_H_
#define ZIMG_GRAPH_IMAGE_FILTER_H_

#include <cstddef>
#include <limits>
#include <utility>
#include "image_buffer.h"

namespace zimg {

enum class PixelType;

namespace graph {

/**
 * Interface for image filters.
 */
class ImageFilter {
public:
	/**
	 * Flags structure.
	 */
	struct filter_flags {
		/**
		 * Filter retains state between invocations and must be invoked on
		 * sequentially increasing line indices.
		 */
		bool has_state : 1;

		/**
		 * Filter produces output lines from input lines of the same index.
		 */
		bool same_row : 1;

		/**
		 * Filter can be applied in-place, destroying the input. If the output type
		 * is wider than the input, the output stride may need to be increased.
		 */
		bool in_place : 1;

		/**
		 * Filter only processes entire rows.
		 */
		bool entire_row : 1;

		/**
		 * Filter only processes entire planes.
		 */
		bool entire_plane : 1;

		/**
		 * Filter processes three planes simultaneously.
		 */
		bool color : 1;
	};

	/**
	 * Filter output format structure.
	 */
	struct image_attributes {
		unsigned width;
		unsigned height;
		PixelType type;
	};

	typedef std::pair<unsigned, unsigned> pair_unsigned;

	/**
	 * Destroy filter.
	 */
	virtual ~ImageFilter() = 0;

	/**
	 * Get the filter flags structure.
	 *
	 * @return flags
	 */
	virtual filter_flags get_flags() const = 0;

	/**
	 * Get the format of the filter output.
	 *
	 * @return attributes
	 */
	virtual image_attributes get_image_attributes() const = 0;

	/**
	 * Get the row range required to produce a given line.
	 *
	 * @param row index
	 * @return range
	 */
	virtual pair_unsigned get_required_row_range(unsigned i) const = 0;

	/**
	 * Get the column range required to produce a given horizontal span.
	 *
	 * @param left left column index
	 * @param right right column index, plus one
	 * @return range
	 */
	virtual pair_unsigned get_required_col_range(unsigned left, unsigned right) const = 0;

	/**
	 * Get the number of lines produced in a single call to {@link process}.
	 *
	 * @return line count, may be {@link BUFFER_MAX}
	 */
	virtual unsigned get_simultaneous_lines() const = 0;

	/**
	 * Get the maximum number of input lines used in any call to {@link process}.
	 *
	 * @return line count, may be {@link BUFFER_MAX}
	 */
	virtual unsigned get_max_buffering() const = 0;

	/**
	 * Get size of the per-frame filter context.
	 *
	 * @return size in bytes
	 * @throw error::OutOfMemory if size exceeds SIZE_MAX
	 */
	virtual size_t get_context_size() const = 0;

	/**
	 * Get the size of the per-call temporary buffer.
	 *
	 * @param left left column index to produce
	 * @param right right column index, plus one
	 * @return size in bytes
	 * @throw error::OutOfMemory if size exceeds SIZE_MAX
	 */
	virtual size_t get_tmp_size(unsigned left, unsigned right) const = 0;

	/**
	 * Initialize per-frame filter context.
	 *
	 * @param ctx context
	 */
	virtual void init_context(void *ctx) const = 0;

	/**
	 * Produce a range of output pixels.
	 *
	 * @param ctx per-frame context
	 * @param src input buffer
	 * @param dst output buffer
	 * @param tmp temporary buffer
	 * @param i row index
	 * @param left left column index
	 * @param right right column index, plus one
	 */
	virtual void process(void *ctx, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, unsigned i, unsigned left, unsigned right) const = 0;
};

/**
 * Default implementation of some {@link ImageFilter} member functions.
 */
class ImageFilterBase : public ImageFilter {
public:
	virtual ~ImageFilterBase() = 0;

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		return{ i, i + 1 };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		return{ left, right };
	}

	unsigned get_simultaneous_lines() const override { return 1; }
	unsigned get_max_buffering() const override { return 1; }

	size_t get_context_size() const override { return 0; }
	size_t get_tmp_size(unsigned, unsigned) const override { return 0; }

	void init_context(void *ctx) const override {}
};

inline ImageFilter::~ImageFilter() = default;

inline ImageFilterBase::~ImageFilterBase() = default;

/**
 * Compare two {@link ImageFilter::image_attributes} structures.
 *
 * @return true if structures are equal, else false
 */
constexpr bool operator==(const ImageFilter::image_attributes &a, const ImageFilter::image_attributes &b) noexcept
{
	return a.width == b.width && a.height == b.height && a.type == b.type;
}

/**
 * @see operator==(const ImageFilter::image_attributes &, const ImageFilter::image_attributes &)
 */
constexpr bool operator!=(const ImageFilter::image_attributes &a, const ImageFilter::image_attributes &b) noexcept
{
	return !(a == b);
}

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_IMAGE_FILTER_H_
