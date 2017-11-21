#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "common/align.h"
#include "common/alloc.h"
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "copy_filter.h"
#include "filtergraph.h"
#include "image_filter.h"

namespace zimg {
namespace graph {
namespace {

class ColorExtendFilter : public CopyFilter {
	bool m_rgb;
public:
	ColorExtendFilter(const image_attributes &attr, bool rgb) :
		CopyFilter{ attr.width, attr.height, attr.type },
		m_rgb{ rgb }
	{}

	filter_flags get_flags() const override
	{
		filter_flags flags = CopyFilter::get_flags();

		flags.in_place = false;
		flags.color = true;

		return flags;
	}

	void process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override
	{
		CopyFilter::process(nullptr, src, dst + 0, nullptr, i, left, right);

		if (m_rgb) {
			CopyFilter::process(nullptr, src, dst + 1, nullptr, i, left, right);
			CopyFilter::process(nullptr, src, dst + 2, nullptr, i, left, right);
		}
	}
};

class ChromaInitializeFilter : public ImageFilterBase {
	image_attributes m_attr;
	unsigned m_subsample_w;
	unsigned m_subsample_h;

	union {
		uint8_t b;
		uint16_t w;
		float f;
	} m_value;

	template <class T>
	void fill(T *ptr, const T &val, unsigned left, unsigned right) const
	{
		std::fill(ptr + left, ptr + right, val);
	}
public:
	ChromaInitializeFilter(image_attributes attr, unsigned subsample_w, unsigned subsample_h, unsigned depth) :
		m_attr{ attr.width >> subsample_w, attr.height >> subsample_h, attr.type },
		m_subsample_w{ subsample_w },
		m_subsample_h{ subsample_h },
		m_value{}
	{
		if (attr.type == PixelType::BYTE)
			m_value.b = static_cast<uint8_t>(1U << (depth - 1));
		else if (attr.type == PixelType::WORD)
			m_value.w = static_cast<uint16_t>(1U << (depth - 1));
		else if (attr.type == PixelType::HALF)
			m_value.w = 0;
		else if (attr.type == PixelType::FLOAT)
			m_value.f = 0.0f;
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = true;
		flags.in_place = true;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return m_attr;
	}

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		return{ i << m_subsample_h, (i + 1) << m_subsample_h };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		return{ left << m_subsample_w, right << m_subsample_w };
	}

	void process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override
	{
		void *dst_p = (*dst)[i];

		if (m_attr.type == PixelType::BYTE)
			fill(static_cast<uint8_t *>(dst_p), m_value.b, left, right);
		else if (m_attr.type == PixelType::WORD || m_attr.type == PixelType::HALF)
			fill(static_cast<uint16_t *>(dst_p), m_value.w, left, right);
		else if (m_attr.type == PixelType::FLOAT)
			fill(static_cast<float *>(dst_p), m_value.f, left, right);
	}
};


enum class ExecutionStrategy {
	LUMA,
	CHROMA,
	COLOR,
};

class ExecutionState {
	struct guard_page {
		static constexpr uint32_t GUARD_VALUE = 0xDEADBEEFUL;

		uint32_t x[AlignmentOf<uint32_t>::value];

		guard_page()
		{
			for (uint32_t &v : x) {
				v = GUARD_VALUE;
			}
		}

		void check() const
		{
			for (uint32_t v : x) {
				zassert_d(v == GUARD_VALUE, "buffer overflow detected");
			}
		}
	};
public:
	struct cache_state {
		ColorImageBuffer<void> buffer;
		bool external;
	};

	struct node_cache_state {
		unsigned cache_pos;
		unsigned source_left;
		unsigned source_right;
	};
private:
	LinearAllocator m_alloc;
	FilterGraph::callback m_unpack_cb;
	FilterGraph::callback m_pack_cb;
	cache_state *m_cache_table;
	node_cache_state *m_node_table;
	void **m_context_table;
	void *m_base;

	guard_page **m_guard;
	size_t m_guard_idx;

	void alloc_guard_page()
	{
		m_guard[m_guard_idx] = m_alloc.allocate_n<guard_page>(1);
		new (m_guard[m_guard_idx]) guard_page{};
		m_guard_idx++;
	}
public:
	static size_t table_size(unsigned num_contexts)
	{
		FakeAllocator alloc;

		alloc.allocate_n<cache_state>(num_contexts);
		alloc.allocate_n<node_cache_state>(num_contexts);
		alloc.allocate_n<void *>(num_contexts);
		alloc.allocate_n<guard_page *>(num_contexts * 8);
		alloc.allocate_n<guard_page>(num_contexts * 8);

		return alloc.count();
	}

	ExecutionState(unsigned num_contexts, void *pool, FilterGraph::callback unpack_cb, FilterGraph::callback pack_cb) :
		m_alloc{ pool },
		m_unpack_cb{ unpack_cb },
		m_pack_cb{ pack_cb },
		m_cache_table{},
		m_node_table{},
		m_context_table{},
		m_base{ pool },
		m_guard{},
		m_guard_idx{}
	{
		m_cache_table = m_alloc.allocate_n<cache_state>(num_contexts);
		std::fill_n(m_cache_table, num_contexts, cache_state{});

		m_node_table = m_alloc.allocate_n<node_cache_state>(num_contexts);
		std::fill_n(m_node_table, num_contexts, node_cache_state{});

		m_context_table = m_alloc.allocate_n<void *>(num_contexts);
		std::fill_n(m_context_table, num_contexts, nullptr);

		m_guard = m_alloc.allocate_n<guard_page *>(num_contexts * 8);
		std::fill_n(m_guard, num_contexts * 8, nullptr);
	}

	void check_guard() const
	{
		for (size_t i = 0; i < m_guard_idx; ++i) {
			m_guard[i]->check();
		}
	}

	void *alloc_context(unsigned id, size_t size)
	{
		zassert_d(!m_context_table[id], "context already allocated");
		alloc_guard_page();
		m_context_table[id] = m_alloc.allocate(size);
		alloc_guard_page();
		return m_context_table[id];
	}

	void alloc_cache(unsigned id, ptrdiff_t stride, unsigned n, unsigned mask, std::array<bool, 3> planes)
	{
		cache_state *cache = m_cache_table + id;
		if (cache->external)
			return;

		for (unsigned p = 0; p < 3; ++p) {
			if (planes[p]) {
				zassert_d(!cache->buffer[p].data(), "cache already allocated");
				alloc_guard_page();
				cache->buffer[p] = { m_alloc.allocate(n * static_cast<size_t>(stride)), stride, mask };
				alloc_guard_page();
			}
		}
	}

	void set_external_buffer(unsigned id, const ColorImageBuffer<void> &buffer)
	{
		cache_state *cache = m_cache_table + id;

		for (unsigned p = 0; p < 3; ++p) {
			if (buffer[p].data()) {
				zassert_d(!cache->buffer[p].data(), "cache already allocated");
				cache->buffer[p] = buffer[p];
			}
		}

		cache->external = true;
	}

	cache_state *get_cache(unsigned id) const { return m_cache_table + id; }
	node_cache_state *get_node_state(unsigned id) const { return m_node_table + id; }
	void *get_context(unsigned id) const { return m_context_table[id]; }
	void *get_tmp() const { return static_cast<char *>(m_base) + m_alloc.count(); }

	FilterGraph::callback get_unpack_cb() const { return m_unpack_cb; }
	FilterGraph::callback get_pack_cb() const { return m_pack_cb; }
};

class GraphNode {
private:
	unsigned m_id;
	unsigned m_cache_id;
	unsigned m_ref_count;
	unsigned m_cache_lines[3];
	bool m_external_buf;
protected:
	explicit GraphNode(unsigned id) :
		m_id{ id },
		m_cache_id{ id },
		m_ref_count{},
		m_cache_lines{},
		m_external_buf{}
	{}

	void set_cache_id(unsigned id)
	{
		zassert_d(m_ref_count == 1, "attempt to set external cache with multiple refs");
		m_cache_id = id;
	}

	void update_cache_state(std::pair<unsigned, unsigned> *cache_state, unsigned n) const
	{
		if (n > cache_state[get_cache_id()].second) {
			unsigned height = get_image_attributes().height;
			unsigned mask = select_zimg_buffer_mask(n);

			if (n >= height || mask == BUFFER_MAX)
				cache_state[get_cache_id()].second = BUFFER_MAX;
			else
				cache_state[get_cache_id()].second = mask + 1;
		}
	}

	void init_cache_context(ExecutionState::node_cache_state *ctx) const
	{
		ctx->cache_pos = 0;
		ctx->source_left = 0;
		ctx->source_right = 0;
	}

	void reset_cache_context(ExecutionState::node_cache_state *ctx) const
	{
		auto attr = get_image_attributes();

		ctx->cache_pos = 0;
		ctx->source_left = attr.width;
		ctx->source_right = 0;
	}
public:
	virtual ~GraphNode() = default;

	unsigned get_id() const { return m_id; }
	unsigned get_cache_id() const { return m_cache_id; }

	void add_ref() { ++m_ref_count; }
	unsigned get_ref() const { return m_ref_count; }

	unsigned get_cache_lines(ExecutionStrategy strategy) const { return m_cache_lines[static_cast<int>(strategy)]; }
	void set_cache_lines(ExecutionStrategy strategy, unsigned n) { m_cache_lines[static_cast<int>(strategy)] = n; }

	bool has_external_buffer() const { return m_external_buf; }
	void set_external_buffer() { m_external_buf = true; }

	virtual ImageFilter::image_attributes get_image_attributes() const = 0;
	virtual ImageFilter::image_attributes get_image_attributes(bool uv) const = 0;

	virtual bool entire_row() const = 0;

	virtual void request_external_cache(unsigned id) = 0;

	virtual void complete() = 0;

	virtual void simulate(std::pair<unsigned, unsigned> *cache_state, unsigned first, unsigned last, bool uv) = 0;

	virtual size_t get_context_size(ExecutionStrategy strategy) const = 0;

	virtual size_t get_tmp_size(unsigned left, unsigned right) const = 0;

	virtual void init_context(ExecutionState *state, ExecutionStrategy strategy) const = 0;

	virtual void reset_context(ExecutionState *state) const = 0;

	virtual void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const = 0;

	virtual void generate_line(ExecutionState *state, unsigned i, bool uv) const = 0;
};

class NullNode final : public GraphNode {
	ImageFilter::image_attributes m_attr;
public:
	NullNode(unsigned id, const ImageFilter::image_attributes &attr) : GraphNode(id), m_attr(attr) {}

	ImageFilter::image_attributes get_image_attributes() const override { return m_attr; }
	ImageFilter::image_attributes get_image_attributes(bool uv) const override { return m_attr; }
	bool entire_row() const override { return false; }
	void request_external_cache(unsigned) override {}
	void complete() override {}
	void simulate(std::pair<unsigned, unsigned> *, unsigned, unsigned, bool) override {}
	size_t get_context_size(ExecutionStrategy) const override { return 0; }
	size_t get_tmp_size(unsigned, unsigned) const override { return 0; }
	void init_context(ExecutionState *, ExecutionStrategy) const override {}
	void reset_context(ExecutionState *) const override {}
	void set_tile_region(ExecutionState *, unsigned, unsigned, bool) const override {}
	void generate_line(ExecutionState *, unsigned, bool) const override {}
};

class SourceNode final : public GraphNode {
	ImageFilter::image_attributes m_attr;
	unsigned m_subsample_w;
	unsigned m_subsample_h;
public:
	SourceNode(unsigned id, unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h) :
		GraphNode(id),
		m_attr{ width, height, type },
		m_subsample_w{ subsample_w },
		m_subsample_h{ subsample_h }
	{}

	ImageFilter::image_attributes get_image_attributes() const override
	{
		return get_image_attributes(false);
	}

	ImageFilter::image_attributes get_image_attributes(bool uv) const override
	{
		auto attr = m_attr;
		attr.width >>= (uv ? m_subsample_w : 0);
		attr.height >>= (uv ? m_subsample_h : 0);
		return attr;
	}

	bool entire_row() const override { return false; }

	void request_external_cache(unsigned id) override
	{
		zassert_d(false, "attempt to set external cache on source node");
	}

	void complete() override {}

	void simulate(std::pair<unsigned, unsigned> *cache_state, unsigned first, unsigned last, bool uv) override
	{
		unsigned step = 1U << m_subsample_h;
		unsigned pos = cache_state[get_id()].first;

		first <<= uv ? m_subsample_h : 0;
		last <<= uv ? m_subsample_h : 0;

		if (pos < last)
			pos = floor_n(last - 1, step) + step;

		cache_state[get_id()].first = pos;
		update_cache_state(cache_state, pos - first);
	}

	size_t get_context_size(ExecutionStrategy) const override { return 0; }
	size_t get_tmp_size(unsigned left, unsigned right) const override { return 0; }

	void init_context(ExecutionState *state, ExecutionStrategy) const override { init_cache_context(state->get_node_state(get_id())); }
	void reset_context(ExecutionState *state) const override { reset_cache_context(state->get_node_state(get_id())); }

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const override
	{
		auto *context = state->get_node_state(get_id());

		left <<= uv ? m_subsample_w : 0;
		right <<= uv ? m_subsample_w : 0;

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);
	}

	void generate_line(ExecutionState *state, unsigned i, bool uv) const override
	{
		auto *context = state->get_node_state(get_id());

		unsigned step = 1U << m_subsample_h;
		unsigned line = i << (uv ? m_subsample_h : 0);
		unsigned pos = context->cache_pos;

		if (line >= pos) {
			if (state->get_unpack_cb()) {
				for (; pos <= line; pos += step) {
					state->get_unpack_cb()(pos, context->source_left, context->source_right);
				}
			} else {
				pos = floor_n(line, step) + step;
			}

			context->cache_pos = pos;
		}
	}
};

class FilterNode : public GraphNode {
protected:
	std::shared_ptr<ImageFilter> m_filter;
	ImageFilter::filter_flags m_flags;
	GraphNode *m_parent;
	unsigned m_step;

	bool is_inplace_capable(const GraphNode *parent) const
	{
		if (!m_flags.in_place || parent->get_ref() > 1 || has_external_buffer() || parent->has_external_buffer())
			return false;

		auto attr = get_image_attributes();
		auto parent_attr = parent->get_image_attributes();

		return attr.width == parent_attr.width && pixel_size(attr.type) == pixel_size(parent_attr.type);
	}

	ptrdiff_t get_cache_stride() const
	{
		auto attr = get_image_attributes();
		return ceil_n(get_image_attributes().width * pixel_size(attr.type), ALIGNMENT);
	}

	unsigned get_real_cache_lines(ExecutionStrategy strategy) const
	{
		return get_cache_lines(strategy) == BUFFER_MAX ? get_image_attributes().height : get_cache_lines(strategy);
	}

	size_t get_cache_size(ExecutionStrategy strategy, unsigned num_planes) const
	{
		checked_size_t rowsize = get_cache_stride();
		checked_size_t size = rowsize * get_real_cache_lines(strategy) * num_planes;
		return size.get();
	}
public:
	FilterNode(unsigned id, std::shared_ptr<ImageFilter> filter, GraphNode *parent) :
		GraphNode(id),
		m_filter{ std::move(filter) },
		m_flags(m_filter->get_flags()),
		m_parent{ parent },
		m_step{ m_filter->get_simultaneous_lines() }
	{}

	ImageFilter::image_attributes get_image_attributes() const override { return m_filter->get_image_attributes(); }
	ImageFilter::image_attributes get_image_attributes(bool) const override { return m_filter->get_image_attributes(); }

	bool entire_row() const override { return m_flags.entire_row || m_parent->entire_row(); }

	void request_external_cache(unsigned id) override
	{
		if (m_parent->get_cache_id() == get_cache_id())
			m_parent->request_external_cache(id);

		set_cache_id(id);
	}

	void complete() override
	{
		if (is_inplace_capable(m_parent))
			m_parent->request_external_cache(get_cache_id());
	}

	void simulate(std::pair<unsigned, unsigned> *cache_state, unsigned first, unsigned last, bool uv) override
	{
		unsigned pos = cache_state[get_id()].first;

		for (; pos < last; pos += m_step) {
			auto range = m_filter->get_required_row_range(pos);
			m_parent->simulate(cache_state, range.first, range.second, uv);
		}

		cache_state[get_id()].first = pos;
		update_cache_state(cache_state, pos - first);
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = m_filter->get_required_col_range(left, right);
		return std::max(m_filter->get_tmp_size(left, right), m_parent->get_tmp_size(range.first, range.second));
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const override
	{
		auto *context = state->get_node_state(get_id());
		auto range = m_filter->get_required_col_range(left, right);

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);

		m_parent->set_tile_region(state, range.first, range.second, uv);
	}
};

class LumaNode final : public FilterNode {
public:
	using FilterNode::FilterNode;

	ImageFilter::image_attributes get_image_attributes(bool uv) const override
	{
		zassert_d(!uv, "request for chroma plane on luma node");
		return FilterNode::get_image_attributes(false);
	}

	void simulate(std::pair<unsigned, unsigned> *cache_state, unsigned first, unsigned last, bool uv) override
	{
		zassert_d(!uv, "request for chroma plane on luma node");
		FilterNode::simulate(cache_state, first, last, false);
	}

	size_t get_context_size(ExecutionStrategy strategy) const override
	{
		if (strategy != ExecutionStrategy::LUMA && strategy != ExecutionStrategy::COLOR)
			return 0;

		FakeAllocator alloc;

		alloc.allocate(m_filter->get_context_size());
		if (get_cache_id() == get_id())
			alloc.allocate(get_cache_size(strategy, 1));

		return alloc.count();
	}

	void init_context(ExecutionState *state, ExecutionStrategy strategy) const override
	{
		std::array<bool, 3> enabled_planes{ { true, false, false } };

		init_cache_context(state->get_node_state(get_id()));
		if (get_cache_id() == get_id())
			state->alloc_cache(get_cache_id(), get_cache_stride(), get_real_cache_lines(strategy), select_zimg_buffer_mask(get_cache_lines(strategy)), enabled_planes);

		void *filter_ctx = state->alloc_context(get_id(), m_filter->get_context_size());
		m_filter->init_context(filter_ctx);
	}

	void reset_context(ExecutionState *state) const override
	{
		reset_cache_context(state->get_node_state(get_id()));
		m_filter->init_context(state->get_context(get_id()));
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const override
	{
		zassert_d(!uv, "request for chroma plane on luma node");
		FilterNode::set_tile_region(state, left, right, false);
	}

	void generate_line(ExecutionState *state, unsigned i, bool uv) const override
	{
		zassert_d(!uv, "request for chroma plane on luma node");

		auto *context = state->get_node_state(get_id());
		unsigned pos = context->cache_pos;

		const ColorImageBuffer<const void> &input_buffer = static_buffer_cast<const void>(state->get_cache(m_parent->get_cache_id())->buffer);
		const ColorImageBuffer<void> &output_buffer = state->get_cache(get_cache_id())->buffer;

		for (; pos <= i; pos += m_step) {
			auto range = m_filter->get_required_row_range(pos);
			zassert_d(range.first < range.second, "bad row range");

			for (unsigned ii = range.first; ii < range.second; ++ii) {
				m_parent->generate_line(state, ii, false);
			}

			m_filter->process(state->get_context(get_id()), input_buffer, output_buffer, state->get_tmp(), pos, context->source_left, context->source_right);
			state->check_guard();
		}
		context->cache_pos = pos;
	}
};

class ChromaNode final : public FilterNode {
	size_t m_filter_ctx_size;
public:
	ChromaNode(unsigned id, std::shared_ptr<ImageFilter> filter, GraphNode *parent) :
		FilterNode(id, std::move(filter), parent),
		m_filter_ctx_size{}
	{
		m_filter_ctx_size = m_filter->get_context_size();
	}

	ImageFilter::image_attributes get_image_attributes(bool uv) const override
	{
		zassert_d(uv, "request for luma plane on chroma node");
		return FilterNode::get_image_attributes(true);
	}

	void simulate(std::pair<unsigned, unsigned> *cache_state, unsigned first, unsigned last, bool uv) override
	{
		zassert_d(uv, "request for luma plane on chroma node");
		FilterNode::simulate(cache_state, first, last, true);
	}

	size_t get_context_size(ExecutionStrategy strategy) const override
	{
		if (strategy != ExecutionStrategy::CHROMA && strategy != ExecutionStrategy::COLOR)
			return 0;

		FakeAllocator alloc;

		alloc.allocate(m_filter->get_context_size());
		alloc.allocate(m_filter->get_context_size());
		if (get_cache_id() == get_id())
			alloc.allocate(get_cache_size(strategy, 2));

		return alloc.count();
	}

	void init_context(ExecutionState *state, ExecutionStrategy strategy) const override
	{
		std::array<bool, 3> enabled_planes{ { false, true, true } };

		init_cache_context(state->get_node_state(get_id()));
		if (get_cache_id() == get_id())
			state->alloc_cache(get_cache_id(), get_cache_stride(), get_real_cache_lines(strategy), select_zimg_buffer_mask(get_cache_lines(strategy)), enabled_planes);

		size_t filter_ctx_size = m_filter->get_context_size();
		void *filter_ctx = state->alloc_context(get_id(), m_filter->get_context_size() * 2);

		m_filter->init_context(filter_ctx);
		m_filter->init_context(static_cast<unsigned char *>(filter_ctx) + filter_ctx_size);
	}

	void reset_context(ExecutionState *state) const override
	{
		size_t filter_ctx_size = m_filter->get_context_size();
		void *filter_ctx = state->get_context(get_id());

		reset_cache_context(state->get_node_state(get_id()));
		m_filter->init_context(filter_ctx);
		m_filter->init_context(static_cast<unsigned char *>(filter_ctx) + filter_ctx_size);
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const override
	{
		zassert_d(uv, "request for luma plane on chroma node");
		FilterNode::set_tile_region(state, left, right, true);
	}

	void generate_line(ExecutionState *state, unsigned i, bool uv) const
	{
		zassert_d(uv, "request for luma plane on chroma node");

		auto *context = state->get_node_state(get_id());
		unsigned pos = context->cache_pos;

		const ColorImageBuffer<const void> &input_buffer = static_buffer_cast<const void>(state->get_cache(m_parent->get_cache_id())->buffer);
		const ColorImageBuffer<void> &output_buffer = state->get_cache(get_cache_id())->buffer;

		void *filter_ctx_u = state->get_context(get_id());
		void *filter_ctx_v = static_cast<unsigned char *>(filter_ctx_u) + m_filter_ctx_size;

		for (; pos <= i; pos += m_step) {
			auto range = m_filter->get_required_row_range(pos);
			zassert_d(range.first < range.second, "bad row range");

			for (unsigned ii = range.first; ii < range.second; ++ii) {
				m_parent->generate_line(state, ii, true);
			}

			m_filter->process(filter_ctx_u, input_buffer + 1, output_buffer + 1, state->get_tmp(), pos, context->source_left, context->source_right);
			state->check_guard();
			m_filter->process(filter_ctx_v, input_buffer + 2, output_buffer + 2, state->get_tmp(), pos, context->source_left, context->source_right);
			state->check_guard();
		}
		context->cache_pos = pos;
	}
};

class ColorNode final : public FilterNode {
	GraphNode *m_parent_uv;
public:
	ColorNode(unsigned id, std::shared_ptr<ImageFilter> filter, GraphNode *parent, GraphNode *parent_uv) :
		FilterNode(id, std::move(filter), parent),
		m_parent_uv{ parent_uv }
	{}

	bool entire_row() const override
	{
		return m_flags.entire_row || m_parent->entire_row() || m_parent_uv->entire_row();
	}

	void request_external_cache(unsigned id) override
	{
		if (m_parent->get_cache_id() == get_cache_id())
			m_parent->request_external_cache(id);
		if (m_parent_uv->get_cache_id() == get_cache_id())
			m_parent_uv->request_external_cache(id);

		set_cache_id(id);
	}

	void complete() override
	{
		if (is_inplace_capable(m_parent) && is_inplace_capable(m_parent_uv)) {
			m_parent->request_external_cache(get_cache_id());
			m_parent_uv->request_external_cache(get_cache_id());
		}
	}

	void simulate(std::pair<unsigned, unsigned> *cache_state, unsigned first, unsigned last, bool uv) override
	{
		unsigned pos = cache_state[get_id()].first;

		for (; pos < last; pos += m_step) {
			auto range = m_filter->get_required_row_range(pos);
			m_parent->simulate(cache_state, range.first, range.second, false);
			m_parent_uv->simulate(cache_state, range.first, range.second, true);
		}

		cache_state[get_id()].first = pos;
		update_cache_state(cache_state, pos - first);
	}

	size_t get_context_size(ExecutionStrategy strategy) const override
	{
		zassert_d(strategy == ExecutionStrategy::COLOR, "can not access channels independently in color node");

		FakeAllocator alloc;

		alloc.allocate(m_filter->get_context_size());
		if (get_cache_id() == get_id())
			alloc.allocate(get_cache_size(strategy, 3));

		return alloc.count();
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = m_filter->get_required_col_range(left, right);
		size_t size = m_filter->get_tmp_size(left, right);
		size = std::max(size, m_parent->get_tmp_size(range.first, range.second));
		size = std::max(size, m_parent_uv->get_tmp_size(range.first, range.second));
		return size;
	}

	void init_context(ExecutionState *state, ExecutionStrategy strategy) const override
	{
		zassert_d(strategy == ExecutionStrategy::COLOR, "can not access channels independently in color node");

		std::array<bool, 3> enabled_planes{ { true, true, true } };

		init_cache_context(state->get_node_state(get_id()));
		if (get_cache_id() == get_id())
			state->alloc_cache(get_cache_id(), get_cache_stride(), get_real_cache_lines(strategy), select_zimg_buffer_mask(get_cache_lines(strategy)), enabled_planes);

		void *filter_ctx = state->alloc_context(get_id(), m_filter->get_context_size());
		m_filter->init_context(filter_ctx);
	}

	void reset_context(ExecutionState *state) const override
	{
		reset_cache_context(state->get_node_state(get_id()));
		m_filter->init_context(state->get_context(get_id()));
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool) const override
	{
		auto *context = state->get_node_state(get_id());
		auto range = m_filter->get_required_col_range(left, right);

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);

		m_parent->set_tile_region(state, range.first, range.second, false);
		m_parent_uv->set_tile_region(state, range.first, range.second, true);
	}

	void generate_line(ExecutionState *state, unsigned i, bool uv) const override
	{
		auto *context = state->get_node_state(get_id());
		unsigned pos = context->cache_pos;

		const ColorImageBuffer<const void> &input_buffer = static_buffer_cast<const void>(state->get_cache(m_parent->get_cache_id())->buffer);
		const ColorImageBuffer<const void> &input_buffer_uv = static_buffer_cast<const void>(state->get_cache(m_parent_uv->get_cache_id())->buffer);
		const ColorImageBuffer<void> &output_buffer = state->get_cache(get_cache_id())->buffer;

		const ColorImageBuffer<const void> *real_input_buffer = &input_buffer;
		ColorImageBuffer<const void> xbuffer;

		if (m_parent->get_cache_id() != m_parent_uv->get_cache_id()) {
			xbuffer[0] = input_buffer[0];
			xbuffer[1] = input_buffer_uv[1];
			xbuffer[2] = input_buffer_uv[2];
			real_input_buffer = &xbuffer;
		}

		for (; pos <= i; pos += m_step) {
			auto range = m_filter->get_required_row_range(pos);
			zassert_d(range.first < range.second, "bad row range");

			for (unsigned ii = range.first; ii < range.second; ++ii) {
				m_parent->generate_line(state, ii, false);
				m_parent_uv->generate_line(state, ii, true);
			}

			m_filter->process(state->get_context(get_id()), *real_input_buffer, output_buffer, state->get_tmp(), pos, context->source_left, context->source_right);
			state->check_guard();
		}
		context->cache_pos = pos;
	}
};

} // namespace


class FilterGraph::impl {
	static constexpr unsigned TILE_WIDTH_MIN = 128;

	std::vector<std::unique_ptr<GraphNode>> m_node_set;
	GraphNode *m_head;
	GraphNode *m_node;
	GraphNode *m_node_uv;
	unsigned m_id_counter;
	unsigned m_input_subsample_w;
	unsigned m_input_subsample_h;
	unsigned m_subsample_w;
	unsigned m_subsample_h;
	unsigned m_tile_width;
	bool m_color_input;
	bool m_color_filter;
	bool m_requires_64b_alignment;
	bool m_is_complete;

	void check_incomplete() const
	{
		if (m_is_complete)
			error::throw_<error::InternalError>("cannot modify completed graph");
	}

	void check_complete() const
	{
		if (!m_is_complete)
			error::throw_<error::InternalError>("cannot query properties on incomplete graph");
	}

	size_t get_tmp_size(ExecutionStrategy strategy, unsigned tile_width) const
	{
		auto attr = m_node->get_image_attributes(false);
		unsigned step = tile_width;

		FakeAllocator alloc;
		size_t tmp_size = 0;

		alloc.allocate(ExecutionState::table_size(m_id_counter));

		for (const auto &node : m_node_set) {
			alloc.allocate(node->get_context_size(strategy));
		}

		for (unsigned j = 0; j < attr.width; j += step) {
			unsigned j_end = std::min(j + step, attr.width);

			if (attr.width - j_end < TILE_WIDTH_MIN) {
				j_end = attr.width;
				step = attr.width - j;
			}

			if (strategy == ExecutionStrategy::LUMA || strategy == ExecutionStrategy::COLOR)
				tmp_size = std::max(tmp_size, m_node->get_tmp_size(j, j_end));
			if (m_node_uv && (strategy == ExecutionStrategy::CHROMA || strategy == ExecutionStrategy::COLOR))
				tmp_size = std::max(tmp_size, m_node_uv->get_tmp_size(j >> m_subsample_w, j_end >> m_subsample_w));
		}
		alloc.allocate(tmp_size);

		return alloc.count();
	}

	size_t get_cache_footprint(ExecutionStrategy strategy) const
	{
		auto input_attr = m_head->get_image_attributes();
		auto output_attr = m_node->get_image_attributes();

		unsigned input_buffering = get_input_buffering(strategy);
		unsigned output_buffering = get_output_buffering(strategy);

		if (input_buffering == BUFFER_MAX)
			input_buffering = input_attr.height;
		if (output_buffering == BUFFER_MAX)
			output_buffering = output_attr.height;

		checked_size_t tmp = get_tmp_size(strategy, output_attr.width);

		if (strategy == ExecutionStrategy::LUMA || strategy == ExecutionStrategy::COLOR) {
			tmp += ceil_n(static_cast<checked_size_t>(input_attr.width) * pixel_size(input_attr.type), ALIGNMENT) * input_buffering;
			tmp += ceil_n(static_cast<checked_size_t>(output_attr.width) * pixel_size(output_attr.type), ALIGNMENT) * output_buffering;
		}
		if (m_color_input && (strategy == ExecutionStrategy::CHROMA || strategy == ExecutionStrategy::COLOR))
			tmp += ceil_n(static_cast<checked_size_t>(input_attr.width >> m_input_subsample_w) * pixel_size(input_attr.type), ALIGNMENT) * (input_buffering >> m_input_subsample_h);
		if (m_node_uv && (strategy == ExecutionStrategy::CHROMA || strategy == ExecutionStrategy::COLOR))
			tmp += ceil_n(static_cast<checked_size_t>(output_attr.width >> m_subsample_w) * pixel_size(output_attr.type), ALIGNMENT) * (output_buffering >> m_subsample_h);

		return tmp.get();
	}

	unsigned get_tile_width(ExecutionStrategy strategy) const
	{
		bool entire_row = m_node->entire_row() || (m_node_uv && m_node_uv->entire_row());
		auto attr = m_node->get_image_attributes();

		if (entire_row)
			return attr.width;
		if (m_tile_width)
			return m_tile_width;

		size_t processor_cache = cpu_cache_size();
		size_t footprint = get_cache_footprint(strategy);

		unsigned tile_width = static_cast<unsigned>(std::lrint(static_cast<double>(attr.width) * processor_cache / footprint));

		if (tile_width > attr.width * 5 / 4)
			tile_width = attr.width;
		else if (tile_width > attr.width / 2)
			tile_width = ceil_n(attr.width / 2, ALIGNMENT);
		else if (tile_width > attr.width / 3)
			tile_width = ceil_n(attr.width / 3, ALIGNMENT);
		else
			tile_width = std::max(floor_n(tile_width, ALIGNMENT), TILE_WIDTH_MIN + 0);

		return tile_width;
	}

	void process_color(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
	{
		ExecutionState state{ m_id_counter, tmp, unpack_cb, pack_cb };
		auto attr = m_node->get_image_attributes(false);
		unsigned h_step = get_tile_width(ExecutionStrategy::COLOR);
		unsigned v_step = 1U << m_subsample_h;

		ColorImageBuffer<void> src_;
		ColorImageBuffer<void> dst_;

		for (unsigned p = 0; p < 3; ++p) {
			src_[p] = { const_cast<void *>(src[p].data()), src[p].stride(), src[p].mask() };
			dst_[p] = dst[p];
		}

		state.set_external_buffer(m_head->get_id(), src_);
		state.set_external_buffer(m_node->get_id(), dst_);
		if (m_node_uv && m_node != m_node_uv)
			state.set_external_buffer(m_node_uv->get_id(), dst_);

		for (const auto &node : m_node_set) {
			node->init_context(&state, ExecutionStrategy::COLOR);
		}

		for (unsigned j = 0; j < attr.width; j += h_step) {
			unsigned j_end = std::min(j + h_step, attr.width);

			if (attr.width - j_end < TILE_WIDTH_MIN) {
				j_end = attr.width;
				h_step = attr.width - j;
			}

			for (const auto &node : m_node_set) {
				node->reset_context(&state);
			}

			m_node->set_tile_region(&state, j, j_end, false);
			if (m_node_uv)
				m_node_uv->set_tile_region(&state, j >> m_subsample_w, j_end >> m_subsample_w, true);

			for (unsigned i = 0; i < attr.height; i += v_step) {
				for (unsigned ii = i; ii < i + v_step; ++ii) {
					m_node->generate_line(&state, ii, false);
				}
				if (m_node_uv)
					m_node_uv->generate_line(&state, i >> m_subsample_h, true);

				if (state.get_pack_cb())
					state.get_pack_cb()(i, j, j_end);
			}
		}
	}

	void process_luma(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp) const
	{
		ExecutionState state{ m_id_counter, tmp, nullptr, nullptr };
		auto attr = m_node->get_image_attributes(false);
		unsigned step = get_tile_width(ExecutionStrategy::LUMA);

		ColorImageBuffer<void> src_;
		ColorImageBuffer<void> dst_;

		for (unsigned p = 0; p < 3; ++p) {
			src_[p] = { const_cast<void *>(src[p].data()), src[p].stride(), src[p].mask() };
			dst_[p] = dst[p];
		}

		state.set_external_buffer(m_head->get_id(), src_);
		state.set_external_buffer(m_node->get_id(), dst_);
		if (m_node_uv && m_node != m_node_uv)
			state.set_external_buffer(m_node_uv->get_id(), dst_);

		for (const auto &node : m_node_set) {
			node->init_context(&state, ExecutionStrategy::LUMA);
		}

		for (unsigned j = 0; j < attr.width; j += step) {
			unsigned j_end = std::min(j + step, attr.width);

			if (attr.width - j_end < TILE_WIDTH_MIN) {
				j_end = attr.width;
				step = attr.width - j;
			}

			for (const auto &node : m_node_set) {
				node->reset_context(&state);
			}

			m_node->set_tile_region(&state, j, j_end, false);

			for (unsigned i = 0; i < attr.height; ++i) {
				m_node->generate_line(&state, i, false);
			}
		}
	}

	void process_chroma(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp) const
	{
		ExecutionState state{ m_id_counter, tmp, nullptr, nullptr };
		auto attr = m_node->get_image_attributes(false);
		unsigned h_step = get_tile_width(ExecutionStrategy::CHROMA);
		unsigned v_step = 1U << m_subsample_h;

		ColorImageBuffer<void> src_;
		ColorImageBuffer<void> dst_;

		for (unsigned p = 0; p < 3; ++p) {
			src_[p] = { const_cast<void *>(src[p].data()), src[p].stride(), src[p].mask() };
			dst_[p] = dst[p];
		}

		state.set_external_buffer(m_head->get_id(), src_);
		state.set_external_buffer(m_node->get_id(), dst_);
		if (m_node_uv && m_node != m_node_uv)
			state.set_external_buffer(m_node_uv->get_id(), dst_);

		for (const auto &node : m_node_set) {
			node->init_context(&state, ExecutionStrategy::CHROMA);
		}

		for (unsigned j = 0; j < attr.width; j += h_step) {
			unsigned j_end = std::min(j + h_step, attr.width);

			if (attr.width - j_end < TILE_WIDTH_MIN) {
				j_end = attr.width;
				h_step = attr.width - j;
			}

			for (const auto &node : m_node_set) {
				node->reset_context(&state);
			}

			m_node_uv->set_tile_region(&state, j >> m_subsample_w, j_end >> m_subsample_w, true);

			for (unsigned i = 0; i < attr.height; i += v_step) {
				m_node_uv->generate_line(&state, i >> m_subsample_h, true);
			}
		}
	}
public:
	impl(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color) :
		m_head{},
		m_node{},
		m_node_uv{},
		m_id_counter{},
		m_input_subsample_w{ subsample_w },
		m_input_subsample_h{ subsample_h },
		m_subsample_w{},
		m_subsample_h{},
		m_tile_width{},
		m_color_input{ color },
		m_color_filter{},
		m_requires_64b_alignment{},
		m_is_complete{}
	{
		zassert_d(width <= pixel_max_width(type), "image stride causes overflow");

		if (!color && (subsample_w || subsample_h))
			error::throw_<error::InternalError>("greyscale images can not be subsampled");
		if (subsample_w > 2 || subsample_h > 2)
			error::throw_<error::InternalError>("subsampling factor must not exceed 4");

		m_node_set.emplace_back(
			ztd::make_unique<SourceNode>(m_id_counter++, width, height, type, subsample_w, subsample_h));
		m_head = m_node_set.back().get();
		m_node = m_head;

		if (color)
			m_node_uv = m_head;
	}

	void attach_filter(std::shared_ptr<ImageFilter> filter)
	{
		check_incomplete();

		ImageFilter::filter_flags flags = filter->get_flags();
		GraphNode *parent = m_node;
		GraphNode *parent_uv = nullptr;

		if (flags.color) {
			if (!m_node_uv)
				error::throw_<error::InternalError>("cannot use color filter in greyscale graph");

			auto attr = m_node->get_image_attributes(false);
			auto attr_uv = m_node_uv->get_image_attributes(true);

			if (attr.width != attr_uv.width || attr.height != attr_uv.height || attr.type != attr_uv.type)
				error::throw_<error::InternalError>("cannot use color filter with mismatching Y and UV format");

			parent_uv = m_node_uv;

			m_node_set.reserve(m_node_set.size() + 1);
			m_node_set.emplace_back(
				ztd::make_unique<ColorNode>(m_id_counter++, std::move(filter), parent, parent_uv));
			m_node = m_node_set.back().get();
			m_node_uv = m_node;

			m_color_filter = true;
		} else {
			m_node_set.reserve(m_node_set.size() + 1);
			m_node_set.emplace_back(ztd::make_unique<LumaNode>(m_id_counter++, std::move(filter), parent));
			m_node = m_node_set.back().get();
		}

		parent->add_ref();
		if (parent_uv && parent_uv != parent)
			parent_uv->add_ref();
	}

	void attach_filter_uv(std::shared_ptr<ImageFilter> filter)
	{
		check_incomplete();

		if (filter->get_flags().color)
			error::throw_<error::InternalError>("cannot use color filter as UV filter");

		GraphNode *parent = m_node_uv;

		m_node_set.reserve(m_node_set.size() + 1);
		m_node_set.emplace_back(ztd::make_unique<ChromaNode>(m_id_counter++, std::move(filter), parent));
		m_node_uv = m_node_set.back().get();

		parent->add_ref();
	}

	void color_to_grey()
	{
		check_incomplete();

		if (!m_node_uv)
			error::throw_<error::InternalError>("cannot remove chroma from greyscale image");

		auto attr = m_node->get_image_attributes();
		attach_filter(ztd::make_unique<CopyFilter>(attr.width, attr.height, attr.type));
		m_node_uv = nullptr;
	}

	void grey_to_color(bool yuv, unsigned subsample_w, unsigned subsample_h, unsigned depth)
	{
		check_incomplete();

		if (m_node_uv)
			error::throw_<error::InternalError>("cannot add chroma to color image");

		auto attr = m_node->get_image_attributes();
		m_node_set.emplace_back(ztd::make_unique<NullNode>(m_id_counter++, attr));
		m_node_uv = m_node_set.back().get();

		attach_filter(ztd::make_unique<ColorExtendFilter>(attr, !yuv));
		if (yuv)
			attach_filter_uv(ztd::make_unique<ChromaInitializeFilter>(attr, subsample_w, subsample_h, depth));
	}

	void set_requires_64b_alignment()
	{
		check_incomplete();
		m_requires_64b_alignment = true;
	}

	void set_tile_width(unsigned tile_width) { m_tile_width = tile_width; }

	void complete()
	{
		check_incomplete();

		auto node_attr = m_node->get_image_attributes(false);
		auto node_attr_uv = m_node_uv ? m_node_uv->get_image_attributes(true) : node_attr;
		unsigned subsample_w = 0;
		unsigned subsample_h = 0;

		for (unsigned ss = 0; ss < 3; ++ss) {
			if (node_attr.width == node_attr_uv.width << ss)
				subsample_w = ss;
			if (node_attr.height == node_attr_uv.height << ss)
				subsample_h = ss;
		}

		if (node_attr.width != node_attr_uv.width << subsample_w)
			error::throw_<error::InternalError>("unsupported horizontal subsampling");
		if (node_attr.height != node_attr_uv.height << subsample_h)
			error::throw_<error::InternalError>("unsupported vertical subsampling");
		if (node_attr.type != node_attr_uv.type)
			error::throw_<error::InternalError>("UV pixel type can not differ");

		// Attach copying filters if the graph is empty.
		if (m_node == m_head || m_node->get_ref())
			attach_filter(ztd::make_unique<CopyFilter>(node_attr.width, node_attr.height, node_attr.type));
		if (m_node_uv && (m_node_uv == m_head || m_node_uv->get_ref()))
			attach_filter_uv(ztd::make_unique<CopyFilter>(node_attr_uv.width, node_attr_uv.height, node_attr_uv.type));

		// Mark the ends of the graph as accessing external memory.
		m_head->set_external_buffer();
		m_node->set_external_buffer();
		if (m_node_uv)
			m_node_uv->set_external_buffer();

		// Finalize node connections.
		for (const auto &node : m_node_set) {
			node->complete();
		}

		// Simulate execution.
		std::vector<std::pair<unsigned, unsigned>> cache_state(m_id_counter);
		for (unsigned i = 0; i < node_attr.height; i += (1U << subsample_h)) {
			m_node->simulate(cache_state.data(), i, i + (1U << subsample_h), false);
			if (m_node_uv)
				m_node_uv->simulate(cache_state.data(), i >> subsample_h, (i >> subsample_h) + 1, true);
		}
		for (const auto &node : m_node_set) {
			node->set_cache_lines(ExecutionStrategy::COLOR, cache_state[node->get_id()].second);
		}

		// Simulate the alternative strategy.
		if (!m_color_filter) {
			cache_state.assign(m_id_counter, {});
			for (unsigned i = 0; i < node_attr.height; ++i) {
				m_node->simulate(cache_state.data(), i, i + 1, false);
			}
			for (const auto &node : m_node_set) {
				node->set_cache_lines(ExecutionStrategy::LUMA, cache_state[node->get_id()].second);
			}
		}
		if (m_node_uv && !m_color_filter) {
			cache_state.assign(m_id_counter, {});
			for (unsigned i = 0; i < node_attr.height; i += (1U << subsample_h)) {
				m_node_uv->simulate(cache_state.data(), i >> subsample_h, (i >> subsample_h) + 1, true);
			}
			for (const auto &node : m_node_set) {
				node->set_cache_lines(ExecutionStrategy::CHROMA, cache_state[node->get_id()].second);
			}
		}

		m_subsample_w = subsample_w;
		m_subsample_h = subsample_h;
		m_is_complete = true;
	}

	size_t get_tmp_size() const
	{
		check_complete();

		size_t tmp_size = get_tmp_size(ExecutionStrategy::COLOR, get_tile_width(ExecutionStrategy::COLOR));

		if (!m_color_filter) {
			tmp_size = std::max(tmp_size, get_tmp_size(ExecutionStrategy::LUMA, get_tile_width(ExecutionStrategy::LUMA)));
			tmp_size = std::max(tmp_size, get_tmp_size(ExecutionStrategy::CHROMA, get_tile_width(ExecutionStrategy::CHROMA)));
		}

		return tmp_size;
	}

	unsigned get_input_buffering(ExecutionStrategy strategy = ExecutionStrategy::COLOR) const
	{
		check_complete();
		return m_head->get_cache_lines(strategy);
	}

	unsigned get_output_buffering(ExecutionStrategy strategy = ExecutionStrategy::COLOR) const
	{
		check_complete();

		unsigned lines = 0;

		if (strategy == ExecutionStrategy::LUMA || strategy == ExecutionStrategy::COLOR)
			lines = m_node->get_cache_lines(strategy);

		if (m_node_uv && (strategy == ExecutionStrategy::CHROMA || strategy == ExecutionStrategy::COLOR)) {
			unsigned lines_uv = m_node_uv->get_cache_lines(strategy);
			lines_uv = (lines_uv == BUFFER_MAX) ? lines_uv : lines_uv << m_subsample_h;
			lines = std::max(lines, lines_uv);
		}

		return lines;
	}

	bool requires_64b_alignment() const
	{
		check_complete();
		return m_requires_64b_alignment;
	}

	unsigned tile_width() const
	{
		check_complete();
		return get_tile_width(ExecutionStrategy::COLOR);
	}

	void process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
	{
		check_complete();

		if (m_color_filter || unpack_cb || pack_cb) {
			process_color(src, dst, tmp, unpack_cb, pack_cb);
		} else {
			process_luma(src, dst, tmp);
			if (m_node_uv)
				process_chroma(src, dst, tmp);
		}
	}
};


FilterGraph::callback::callback(std::nullptr_t) : m_func{}, m_user{} {}

FilterGraph::callback::callback(func_type func, void *user) : m_func{ func }, m_user{ user } {}

FilterGraph::callback::operator bool() const { return m_func != nullptr; }

void FilterGraph::callback::operator()(unsigned i, unsigned left, unsigned right) const
{
	int ret;

	try {
		ret = m_func(m_user, i, left, right);
	} catch (...) {
		ret = 1;
		zassert_d(false, "user callback must not throw");
	}

	if (ret)
		error::throw_<error::UserCallbackFailed>("user callback failed");
}


FilterGraph::FilterGraph(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color) :
	m_impl{ ztd::make_unique<impl>(width, height, type, subsample_w, subsample_h, color) }
{}

FilterGraph::FilterGraph(FilterGraph &&other) noexcept = default;

FilterGraph::~FilterGraph() = default;

FilterGraph &FilterGraph::operator=(FilterGraph &&other) noexcept = default;

void FilterGraph::attach_filter(std::shared_ptr<ImageFilter> filter)
{
	get_impl()->attach_filter(std::move(filter));
}

void FilterGraph::attach_filter_uv(std::shared_ptr<ImageFilter> filter)
{
	get_impl()->attach_filter_uv(std::move(filter));
}

void FilterGraph::color_to_grey()
{
	get_impl()->color_to_grey();
}

void FilterGraph::grey_to_color(bool yuv, unsigned subsample_w, unsigned subsample_h, unsigned depth)
{
	get_impl()->grey_to_color(yuv, subsample_w, subsample_h, depth);
}

void FilterGraph::set_requires_64b_alignment()
{
	get_impl()->set_requires_64b_alignment();
}

void FilterGraph::set_tile_width(unsigned tile_width)
{
	get_impl()->set_tile_width(tile_width);
}

void FilterGraph::complete()
{
	get_impl()->complete();
}

size_t FilterGraph::get_tmp_size() const
{
	return get_impl()->get_tmp_size();
}

unsigned FilterGraph::get_input_buffering() const
{
	return get_impl()->get_input_buffering();
}

unsigned FilterGraph::get_output_buffering() const
{
	return get_impl()->get_output_buffering();
}

bool FilterGraph::requires_64b_alignment() const
{
	return get_impl()->requires_64b_alignment();
}

unsigned FilterGraph::tile_width() const
{
	return get_impl()->tile_width();
}

void FilterGraph::process(const ImageBuffer<const void> *src, const ImageBuffer<void> *dst, void *tmp, callback unpack_cb, callback pack_cb) const
{
	get_impl()->process(src, dst, tmp, unpack_cb, pack_cb);
}

} // namespace graph
} // namespace zimg
