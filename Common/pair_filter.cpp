#include <algorithm>
#include "align.h"
#include "alloc.h"
#include "except.h"
#include "pair_filter.h"
#include "pixel.h"

namespace zimg {;

struct PairFilter::cache_context {
	void *first_ctx;
	void *second_ctx;
	void *cache_plane[3];
	unsigned cache_line_pos;
	unsigned cache_mask;
	unsigned col_left;
	unsigned col_right;
};

PairFilter::PairFilter(IZimgFilter *first, IZimgFilter *second) :
	m_first_flags{},
	m_second_flags{},
	m_first_attr{},
	m_second_attr{},
	m_first_step{},
	m_second_step{},
	m_second_buffering{},
	m_has_state{},
	m_in_place{},
	m_color{}
{
	m_first_flags = first->get_flags();
	m_second_flags = second->get_flags();
	m_first_attr = first->get_image_attributes();
	m_second_attr = second->get_image_attributes();

	if (m_first_flags.color != m_second_flags.color)
		throw ZimgLogicError{ "filter pair must be both color or both grey" };

	m_first_step = first->get_simultaneous_lines();
	m_second_step = second->get_simultaneous_lines();
	m_second_buffering = second->get_max_buffering();

	m_has_state = m_first_flags.has_state || m_second_flags.has_state ||
	              !m_first_flags.same_row || !m_second_flags.same_row ||
	              first->get_simultaneous_lines() != second->get_simultaneous_lines();
	m_in_place = m_first_flags.in_place && m_second_flags.in_place && m_first_attr.type == m_second_attr.type;
	m_color = !!m_first_flags.color;

	m_first.reset(first);
	m_second.reset(second);
}

ptrdiff_t PairFilter::get_cache_stride() const
{
	return align(m_first_attr.width * pixel_size(m_first_attr.type), ALIGNMENT);
}

unsigned PairFilter::get_cache_line_count() const
{
	if (m_second_flags.in_place)
		return 0;
	else if (m_first_flags.entire_plane || m_second_flags.entire_plane)
		return m_first_attr.height;
	else if (m_second_flags.same_row && m_first_step == m_second_step)
		return m_second_buffering;
	else
		return m_first_step + m_second_buffering - 1;
}

size_t PairFilter::get_cache_size_one_plane() const
{
	return get_cache_line_count() * get_cache_stride();
}

unsigned PairFilter::get_num_planes() const
{
	return m_color ? 3 : 1;
}

ZimgFilterFlags PairFilter::get_flags() const
{
	ZimgFilterFlags flags{};

	flags.has_state = m_has_state;
	flags.same_row = m_first_flags.same_row && m_second_flags.same_row;
	flags.in_place = m_in_place;
	flags.entire_row = m_first_flags.entire_row || m_second_flags.entire_row;
	flags.entire_plane = m_second_flags.entire_plane;
	flags.color = m_color;

	return flags;
}

IZimgFilter::image_attributes PairFilter::get_image_attributes() const
{
	return m_second_attr;
}

IZimgFilter::pair_unsigned PairFilter::get_required_row_range(unsigned i) const
{
	auto second_range = m_second->get_required_row_range(i);
	auto top_range = m_first->get_required_row_range(mod(second_range.first, m_first_step));
	auto bot_range = m_first->get_required_row_range(mod(second_range.second - 1, m_first_step));

	return{ top_range.first, bot_range.second };
}

IZimgFilter::pair_unsigned PairFilter::get_required_col_range(unsigned left, unsigned right) const
{
	auto second_range = m_second->get_required_col_range(left, right);
	auto first_range = m_first->get_required_col_range(second_range.first, second_range.second);

	return first_range;
}

unsigned PairFilter::get_simultaneous_lines() const
{
	return m_second->get_simultaneous_lines();
}

unsigned PairFilter::get_max_buffering() const
{
	unsigned buffering = 0;

	if (m_first_flags.entire_plane || m_second_flags.entire_plane)
		return -1;

	for (unsigned i = 0; i < m_second_attr.height; i += m_second_step) {
		auto range = get_required_row_range(i);
		buffering = std::max(buffering, range.second - range.first);
	}

	return buffering;
}

size_t PairFilter::get_context_size() const
{
	FakeAllocator alloc{};

	size_t first_context_size = m_first->get_context_size();
	size_t second_context_size = m_second->get_context_size();
	size_t cache_size_one_plane = get_cache_size_one_plane();

	alloc.allocate_n<cache_context>(1);

	alloc.allocate(first_context_size);
	alloc.allocate(second_context_size);

	for (unsigned p = 0; p < get_num_planes(); ++p) {
		alloc.allocate(cache_size_one_plane);
	}

	return alloc.count();
}

size_t PairFilter::get_tmp_size(unsigned left, unsigned right) const
{
	auto range = m_second->get_required_col_range(left, right);

	size_t first_tmp_size = m_first->get_tmp_size(range.first, range.second);
	size_t second_tmp_size = m_second->get_tmp_size(left, right);

	return std::max(first_tmp_size, second_tmp_size);
}

void PairFilter::init_context(void *ctx) const
{
	LinearAllocator alloc{ ctx };

	size_t first_context_size = m_first->get_context_size();
	size_t second_context_size = m_second->get_context_size();
	size_t cache_size_one_plane = get_cache_size_one_plane();

	cache_context *cache = alloc.allocate_n<cache_context>(1);
	new (cache) cache_context{};

	cache->first_ctx = alloc.allocate(first_context_size);
	cache->second_ctx = alloc.allocate(second_context_size);

	m_first->init_context(cache->first_ctx);
	m_second->init_context(cache->second_ctx);

	for (unsigned p = 0; p < get_num_planes(); ++p) {
		cache->cache_plane[p] = alloc.allocate(cache_size_one_plane);
	}

	cache->cache_line_pos = 0;
	cache->cache_mask = select_zimg_buffer_mask(get_cache_line_count());
	cache->col_left = 0;
	cache->col_right = 0;
}

void PairFilter::process(void *ctx, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const
{
	cache_context *cache = static_cast<cache_context *>(ctx);
	auto row_range = m_second->get_required_row_range(i);
	auto col_range = m_second->get_required_col_range(left, right);
	ptrdiff_t cache_stride = get_cache_stride();
	ZimgImageBuffer cache_buf{};

	if (left != cache->col_left || right != cache->col_right) {
		cache->col_left = left;
		cache->col_right = right;
		cache->cache_line_pos = m_first_flags.has_state ? 0 : mod(row_range.first, m_first_step);
	}

	if (m_second_flags.in_place) {
		cache_buf = dst;
	} else {
		for (unsigned p = 0; p < get_num_planes(); ++p) {
			cache_buf.data[p] = cache->cache_plane[p];
			cache_buf.stride[p] = cache_stride;
			cache_buf.mask[p] = cache->cache_mask;
		}
	}

	for (; cache->cache_line_pos < row_range.second; cache->cache_line_pos += m_first_step) {
		m_first->process(cache->first_ctx, src, cache_buf, tmp, cache->cache_line_pos, col_range.first, col_range.second);
	}
	m_second->process(cache->second_ctx, cache_buf, dst, tmp, i, left, right);
}

} // namespace zimg
