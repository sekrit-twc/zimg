#include <algorithm>
#include "Common/align.h"
#include "Common/alloc.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "resize2.h"
#include "resize_impl2.h"

namespace zimg {;
namespace resize {;

namespace {;

bool resize_h_first(double xscale, double yscale)
{
	double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
	double v_first_cost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;

	return h_first_cost < v_first_cost;
}

class PairFilter : public IZimgFilter {
	struct cache_context {
		void *first_ctx[3];
		void *second_ctx[3];
		void *cache_mem[3];
		unsigned cache_pos;
		unsigned cache_mask;
	};

	std::unique_ptr<IZimgFilter> m_first;
	std::unique_ptr<IZimgFilter> m_second;

	ZimgFilterFlags m_first_flags;
	ZimgFilterFlags m_second_flags;

	PixelType m_first_type;
	PixelType m_second_type;
	unsigned m_first_width;
	unsigned m_first_height;
	unsigned m_second_width;
	unsigned m_second_height;

	ptrdiff_t get_cache_stride() const
	{
		return align(m_first_width * pixel_size(m_first_type), ALIGNMENT);
	}

	unsigned get_cache_lines() const
	{
		unsigned first_simultaneous_lines = m_first->get_simultaneous_lines();
		unsigned second_buffering = m_second->get_max_buffering();

		if (m_first_flags.entire_plane || m_second_flags.entire_plane)
			return m_first_height;
		else if (m_second_flags.same_row && first_simultaneous_lines == second_buffering)
			return second_buffering;
		else
			return first_simultaneous_lines + second_buffering - 1;
	}

	size_t get_cache_plane_size() const
	{
		return (size_t)get_cache_lines() * get_cache_stride();
	}
public:
	PairFilter(IZimgFilter *first, IZimgFilter *second, PixelType first_type, PixelType second_type,
	             unsigned first_width, unsigned first_height, unsigned second_width, unsigned second_height) :
		m_first{ first },
		m_second{ second },
		m_first_type{ first_type },
		m_second_type{ second_type },
		m_first_width{ first_width },
		m_first_height{ first_height },
		m_second_width{ second_width },
		m_second_height{ second_height }
	{
		try {
			m_first_flags = m_first->get_flags();
			m_second_flags = m_second->get_flags();
		} catch (...) {
			m_first.release();
			m_second.release();
			throw;
		}
	}

	ZimgFilterFlags get_flags() const override
	{
		ZimgFilterFlags flags{};

		flags.has_state = m_first_flags.has_state || m_second_flags.has_state ||
		                  !(m_first_flags.same_row && m_second_flags.same_row && m_first->get_simultaneous_lines() == m_second->get_simultaneous_lines());
		flags.same_row = m_first_flags.same_row && m_second_flags.same_row;
		flags.in_place = m_first_flags.in_place && m_second_flags.in_place && m_first_type == m_second_type;
		flags.entire_row = m_first_flags.entire_row || m_second_flags.entire_row;
		flags.entire_plane = m_second_flags.entire_plane;
		flags.color = m_first_flags.color || m_second_flags.color;

		return flags;
	}

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		pair_unsigned second_range = m_second->get_required_row_range(i);
		pair_unsigned top_range = m_first->get_required_row_range(second_range.first);
		pair_unsigned bot_range = m_first->get_required_row_range(second_range.second);

		return{ top_range.first, bot_range.second };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		pair_unsigned second_range = m_second->get_required_col_range(left, right);
		pair_unsigned first_range = m_first->get_required_col_range(second_range.first, second_range.second);

		return first_range;
	}

	unsigned get_simultaneous_lines() const override
	{
		return m_second->get_simultaneous_lines();
	}

	unsigned get_max_buffering() const override
	{
		unsigned first_simultaneous_lines = m_first->get_simultaneous_lines();
		unsigned second_simultaneous_lines = m_second->get_simultaneous_lines();
		unsigned buffering = 0;

		if (m_first_flags.entire_plane || m_second_flags.entire_plane)
			return -1;

		for (unsigned i = 0; i < m_second_height; i += second_simultaneous_lines) {
			unsigned range_end = std::min(i + second_simultaneous_lines, m_second_height);

			pair_unsigned second_top_range = m_second->get_required_row_range(i);
			pair_unsigned second_bot_range = m_second->get_required_row_range(range_end - 1);

			pair_unsigned first_top_range = m_first->get_required_row_range(mod(second_top_range.first, first_simultaneous_lines));
			pair_unsigned first_bot_range = m_first->get_required_row_range(mod(second_bot_range.second - 1, first_simultaneous_lines));

			buffering = std::max(buffering, first_bot_range.second - first_top_range.first);
		}

		return buffering;
	}

	size_t get_context_size() const override
	{
		bool color = m_first_flags.color || m_second_flags.color;

		size_t first_context_size = align(m_first->get_context_size(), ALIGNMENT);
		size_t second_context_size = align(m_second->get_context_size(), ALIGNMENT);
		size_t cache_size = get_cache_plane_size();
		size_t cache_context_size = align(sizeof(cache_context), ALIGNMENT);

		if (!m_first_flags.color && color)
			first_context_size *= 3;
		if (!m_second_flags.color && color)
			second_context_size *= 3;
		if (color)
			cache_size *= 3;

		return first_context_size + second_context_size + cache_size + cache_context_size;
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		pair_unsigned second_range = m_second->get_required_col_range(left, right);
		size_t first_tmp_size = m_first->get_tmp_size(second_range.first, second_range.second);
		size_t second_tmp_size = m_second->get_tmp_size(left, right);

		return std::max(first_tmp_size, second_tmp_size);
	}

	void init_context(void *ctx) const override
	{
		LinearAllocator alloc{ ctx };
		bool color = m_first_flags.color || m_second_flags.color;

		size_t first_context_size = m_first->get_context_size();
		size_t second_context_size = m_second->get_context_size();
		size_t cache_size = get_cache_plane_size();

		cache_context *cache_ctx = alloc.allocate_n<cache_context>(1);
		*cache_ctx = {};

		cache_ctx->cache_pos = 0;
		cache_ctx->cache_mask = select_zimg_buffer_mask(get_cache_lines());

		for (unsigned p = 0; p < ((color && !m_first_flags.color) ? 3U : 1U); ++p) {
			void *ptr = alloc.allocate(first_context_size);
			m_first->init_context(ptr);
			cache_ctx->first_ctx[p] = ptr;
		}
		for (unsigned p = 0; p < ((color & !m_second_flags.color) ? 3U : 1U); ++p) {
			void *ptr = alloc.allocate(second_context_size);
			m_second->init_context(ptr);
			cache_ctx->second_ctx[p] = ptr;
		}
		for (unsigned p = 0; p < (color ? 3U : 1U); ++p) {
			void *ptr = alloc.allocate(cache_size);
			cache_ctx->cache_mem[p] = ptr;
		}
	}

	void process(void *ctx, const ZimgImageBuffer *src, const ZimgImageBuffer *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		cache_context *cache_ctx = reinterpret_cast<cache_context *>(ctx);
		pair_unsigned req_row_range = m_second->get_required_row_range(i);
		pair_unsigned req_col_range = m_second->get_required_col_range(left, right);
		unsigned first_simultaneous_lines = m_first->get_simultaneous_lines();
		bool color = m_first_flags.color || m_second_flags.color;

		ptrdiff_t cache_stride = get_cache_stride();

		while (cache_ctx->cache_pos < req_row_range.second) {
			if (color && !m_first_flags.color) {
				for (unsigned p = 0; p < 3; ++p) {
					ZimgImageBuffer src_plane(*src);
					ZimgImageBuffer cache_buf{};

					src_plane.data[0] = src->data[p];
					src_plane.stride[0] = src->stride[p];
					src_plane.mask[0] = src->mask[p];

					cache_buf.data[0] = cache_ctx->cache_mem[p];
					cache_buf.stride[0] = cache_stride;
					cache_buf.mask[0] = cache_ctx->cache_mask;

					m_first->process(cache_ctx->first_ctx[p], &src_plane, &cache_buf, tmp, cache_ctx->cache_pos, req_col_range.first, req_col_range.second);
				}
			} else {
				ZimgImageBuffer cache_buf{};

				for (unsigned p = 0; p < 3; ++p) {
					cache_buf.data[p] = cache_ctx->cache_mem[p];
					cache_buf.stride[p] = cache_stride;
					cache_buf.mask[p] = cache_ctx->cache_mask;
				}

				m_first->process(cache_ctx->first_ctx[0], src, &cache_buf, tmp, cache_ctx->cache_pos, req_col_range.first, req_col_range.second);
			}

			cache_ctx->cache_pos += first_simultaneous_lines;
		}

		if (color && !m_second_flags.color) {
			for (unsigned p = 0; p < 3; ++p) {
				ZimgImageBuffer cache_buf{};
				ZimgImageBuffer dst_plane(*dst);

				cache_buf.data[0] = cache_ctx->cache_mem[p];
				cache_buf.stride[0] = cache_stride;
				cache_buf.mask[0] = cache_ctx->cache_mask;

				dst_plane.data[0] = dst->data[p];
				dst_plane.stride[0] = dst->stride[p];
				dst_plane.mask[0] = dst->mask[p];

				m_second->process(cache_ctx->second_ctx[p], &cache_buf, &dst_plane, tmp, i, left, right);
			}
		} else {
			ZimgImageBuffer cache_buf{};

			for (unsigned p = 0; p < 3; ++p) {
				cache_buf.data[p] = cache_ctx->cache_mem[p];
				cache_buf.stride[p] = cache_stride;
				cache_buf.mask[p] = cache_ctx->cache_mask;
			}

			m_second->process(cache_ctx->second_ctx[0], &cache_buf, dst, tmp, i, left, right);
		}
	}
};

class CopyFilter : public ZimgFilter {
	PixelType m_type;
public:
	explicit CopyFilter(PixelType type) :
		m_type{ type }
	{
	}

	ZimgFilterFlags get_flags() const override
	{
		ZimgFilterFlags flags{};

		flags.same_row = 1;
		flags.in_place = 1;

		return flags;
	}

	void process(void *, const ZimgImageBuffer *src, const ZimgImageBuffer *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		LineBuffer<void> src_buf{ reinterpret_cast<char *>(src->data[0]) + left * pixel_size(m_type), right - left, (unsigned)src->stride[0], src->mask[0] };
		LineBuffer<void> dst_buf{ reinterpret_cast<char *>(dst->data[0]) + left * pixel_size(m_type), right - left, (unsigned)dst->stride[0], dst->mask[0] };

		copy_buffer_lines(src_buf, dst_buf, pixel_size(m_type) * (right - left), i, i + 1);
	}
};

} // namespace


Resize2::Resize2(const Filter &filter, PixelType type, int src_width, int src_height, int dst_width, int dst_height,
                 double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu)
try
{
	bool skip_h = (src_width == dst_width && shift_w == 0 && subwidth == src_width);
	bool skip_v = (src_height == dst_height && shift_h == 0 && subheight == src_height);

	if (skip_h && skip_v) {
		m_impl.reset(new CopyFilter{ type });
	} else if (skip_h) {
		m_impl.reset(create_resize_impl2(filter, type, false, src_height, dst_height, dst_height, shift_h, subheight, cpu));
	} else if (skip_v) {
		m_impl.reset(create_resize_impl2(filter, type, true, src_width, dst_width, dst_height, shift_w, subwidth, cpu));
	} else {
		bool h_first = resize_h_first((double)dst_width / src_width, (double)dst_height / src_height);
		std::unique_ptr<IZimgFilter> stage1;
		std::unique_ptr<IZimgFilter> stage2;
		unsigned tmp_width;
		unsigned tmp_height;

		if (h_first) {
			stage1.reset(create_resize_impl2(filter, type, true, src_width, dst_width, src_height, shift_w, subwidth, cpu));
			stage2.reset(create_resize_impl2(filter, type, false, src_height, dst_height, dst_height, shift_h, subheight, cpu));

			tmp_width = dst_width;
			tmp_height = src_height;
		} else {
			stage1.reset(create_resize_impl2(filter, type, false, src_height, dst_height, dst_height, shift_h, subheight, cpu));
			stage2.reset(create_resize_impl2(filter, type, true, src_width, dst_width, dst_height, shift_w, subwidth, cpu));

			tmp_width = src_width;
			tmp_height = dst_height;
		}

		m_impl.reset(new PairFilter(stage1.get(), stage2.get(), type, type, tmp_width, tmp_height, dst_width, dst_height));
		stage1.release();
		stage2.release();
	}
} catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

ZimgFilterFlags Resize2::get_flags() const
{
	return m_impl->get_flags();
}

IZimgFilter::pair_unsigned Resize2::get_required_row_range(unsigned i) const
{
	return m_impl->get_required_row_range(i);
}

IZimgFilter::pair_unsigned Resize2::get_required_col_range(unsigned left, unsigned right) const
{
	return m_impl->get_required_col_range(left, right);
}

unsigned Resize2::get_simultaneous_lines() const
{
	return m_impl->get_simultaneous_lines();
}

unsigned Resize2::get_max_buffering() const
{
	return m_impl->get_max_buffering();
}

size_t Resize2::get_context_size() const
{
	return m_impl->get_context_size();
}

size_t Resize2::get_tmp_size(unsigned left, unsigned right) const
{
	return m_impl->get_tmp_size(left, right);
}

void Resize2::init_context(void *ctx) const
{
	m_impl->init_context(ctx);
}

void Resize2::process(void *ctx, const ZimgImageBuffer *src, const ZimgImageBuffer *dst, void *tmp, unsigned i, unsigned left, unsigned right) const
{
	m_impl->process(ctx, src, dst, tmp, i, left, right);
}

} // namespace resize
} // namespace zimg
