#include <algorithm>
#include <cstdint>
#include <vector>
#include "align.h"
#include "alloc.h"
#include "copy_filter.h"
#include "except.h"
#include "filtergraph.h"
#include "linebuffer.h"
#include "pixel.h"
#include "zassert.h"
#include "zfilter.h"

namespace zimg {;
namespace {;

class SimulationState {
	std::vector<unsigned> m_cache_pos;
public:
	explicit SimulationState(unsigned size) : m_cache_pos(size)
	{
	}

	unsigned &pos(unsigned id)
	{
		return m_cache_pos[id];
	}
};

class ExecutionState {
	LinearAllocator m_alloc;
	const ZimgImageBufferConst *m_src_buf;
	const ZimgImageBuffer *m_dst_buf;
	FilterGraph::callback m_unpack_cb;
	FilterGraph::callback m_pack_cb;
	void **m_context_table;
	void *m_base;
public:
	ExecutionState(unsigned id_counter, const ZimgImageBufferConst &src_buf, const ZimgImageBuffer &dst_buf, void *pool,
	               FilterGraph::callback unpack_cb, FilterGraph::callback pack_cb) :
		m_alloc{ pool },
		m_src_buf{ &src_buf },
		m_dst_buf{ &dst_buf },
		m_unpack_cb{ unpack_cb },
		m_pack_cb{ pack_cb },
		m_context_table{},
		m_base{ pool }
	{
		m_context_table = m_alloc.allocate_n<void *>(id_counter);
		std::fill_n(m_context_table, id_counter, nullptr);
	}

	void *alloc_context(unsigned id, size_t size)
	{
		_zassert(!m_context_table[id], "context already allocated");

		m_context_table[id] = m_alloc.allocate(size);
		return m_context_table[id];
	}

	void *get_context(unsigned id) const
	{
		return m_context_table[id];
	}

	const ZimgImageBufferConst &get_input_buffer() const
	{
		return *m_src_buf;
	}

	const ZimgImageBuffer &get_output_buffer() const
	{
		return *m_dst_buf;
	}

	void *get_tmp() const
	{
		return reinterpret_cast<char *>(m_base) + m_alloc.count();
	}

	FilterGraph::callback get_unpack_cb() const
	{
		return m_unpack_cb;
	}

	FilterGraph::callback get_pack_cb() const
	{
		return m_pack_cb;
	}

	static size_t context_table_size(unsigned id_counter)
	{
		return align(sizeof(void *) * id_counter, ALIGNMENT);
	}
};


class GraphNode {
	struct source_tag {};
	struct filter_tag {};
	struct filter_uv_tag {};

	struct node_context {
		static const uint64_t GUARD_PATTERN = (((uint64_t)0xDEADBEEF) << 32) | 0xDEADBEEF;

		const uint64_t guard_pattern = GUARD_PATTERN;
		ZimgImageBuffer cache_buf;
		unsigned cache_pos;
		unsigned source_left;
		unsigned source_right;
		void *filter_ctx;
		void *filter_ctx2;

		void assert_guard_pattern() const
		{
			_zassert(guard_pattern == GUARD_PATTERN, "buffer overflow detected");
		}
	};

	union data_union {
		struct {
			unsigned width;
			unsigned height;
			PixelType type;
			unsigned subsample_w;
			unsigned subsample_h;
			bool color;
		} source_info;

		struct {
			GraphNode *parent;
			GraphNode *parent_uv;
			ZimgFilterFlags flags;
			unsigned step;
			bool is_uv;
		} node_info;

		data_union(source_tag) : source_info{} {};
		data_union(filter_tag) : node_info{} {};
		data_union(filter_uv_tag) : node_info{} {};
	} m_data;
public:
	static const source_tag SOURCE;
	static const filter_tag FILTER;
	static const filter_uv_tag FILTER_UV;
private:
	std::unique_ptr<IZimgFilter> m_filter;
	unsigned m_cache_lines;
	unsigned m_ref_count;
	unsigned m_id;
	bool m_is_source;

	unsigned get_num_planes() const
	{
		if (m_is_source)
			return m_data.source_info.color ? 3 : 1;
		else if (m_data.node_info.is_uv)
			return 2;
		else
			return m_data.node_info.flags.color ? 3 : 1;
	}

	unsigned get_real_cache_lines() const
	{
		return m_cache_lines == (unsigned)-1 ? get_image_attributes().height : m_cache_lines;
	}

	ptrdiff_t get_cache_stride() const
	{
		auto attr = get_image_attributes();
		return align(attr.width * pixel_size(attr.type), ALIGNMENT);
	}

	void set_cache_lines(unsigned n)
	{
		if (n > m_cache_lines) {
			unsigned height = m_is_source ? m_data.source_info.height : m_filter->get_image_attributes().height;
			m_cache_lines = n >= height ? -1 : select_zimg_buffer_mask(n) + 1;
		}
	}

	void simulate_source(SimulationState *sim, unsigned first, unsigned last, bool uv)
	{
		unsigned step = 1 << m_data.source_info.subsample_h;
		unsigned pos = sim->pos(m_id);

		first <<= uv ? m_data.source_info.subsample_h : 0;
		last <<= uv ? m_data.source_info.subsample_h : 0;

		if (pos < last)
			pos = mod(last - 1, step) + step;

		sim->pos(m_id) = pos;
		set_cache_lines(pos - first);
	}

	void simulate_node_uv(SimulationState *sim, unsigned first, unsigned last)
	{
		unsigned pos = sim->pos(m_id);

		for (; pos < last; pos += m_data.node_info.step) {
			auto range = m_filter->get_required_row_range(pos);

			m_data.node_info.parent->simulate(sim, range.first, range.second, true);
		}

		sim->pos(m_id) = pos;
		set_cache_lines(pos - first);
	}

	void simulate_node(SimulationState *sim, unsigned first, unsigned last)
	{
		unsigned pos = sim->pos(m_id);

		for (; pos < last; pos += m_data.node_info.step) {
			auto range = m_filter->get_required_row_range(pos);

			m_data.node_info.parent->simulate(sim, range.first, range.second, false);

			if (m_data.node_info.parent_uv)
				m_data.node_info.parent_uv->simulate(sim, range.first, range.second, true);
		}

		sim->pos(m_id) = pos;
		set_cache_lines(pos - first);
	}

	void init_context_node_uv(LinearAllocator &alloc, node_context *context) const
	{
		size_t filter_context_size = m_filter->get_context_size();
		ptrdiff_t stride = get_cache_stride();
		unsigned cache_lines = get_real_cache_lines();

		context->filter_ctx = alloc.allocate(filter_context_size);
		context->filter_ctx2 = alloc.allocate(filter_context_size);

		for (unsigned p = 1; p < 3; ++p) {
			context->cache_buf.data[p] = alloc.allocate((size_t)cache_lines * stride);
			context->cache_buf.stride[p] = stride;
			context->cache_buf.mask[p] = select_zimg_buffer_mask(m_cache_lines);
		}
	}

	void init_context_node(LinearAllocator &alloc, node_context *context) const
	{
		ptrdiff_t stride = get_cache_stride();
		unsigned cache_lines = get_real_cache_lines();

		context->filter_ctx = alloc.allocate(m_filter->get_context_size());

		for (unsigned p = 0; p < get_num_planes(); ++p) {
			context->cache_buf.data[p] = alloc.allocate((size_t)cache_lines * stride);
			context->cache_buf.stride[p] = stride;
			context->cache_buf.mask[p] = select_zimg_buffer_mask(m_cache_lines);
		}
	}

	void set_tile_region_source(ExecutionState *state, unsigned left, unsigned right, bool uv) const
	{
		node_context *context = reinterpret_cast<node_context *>(state->get_context(m_id));
		context->assert_guard_pattern();

		if (uv) {
			left <<= m_data.source_info.subsample_w;
			right <<= m_data.source_info.subsample_w;
		}

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);
	}

	void set_tile_region_node_uv(ExecutionState *state, unsigned left, unsigned right) const
	{
		node_context *context = reinterpret_cast<node_context *>(state->get_context(m_id));
		context->assert_guard_pattern();

		auto range = m_filter->get_required_col_range(left, right);

		m_data.node_info.parent->set_tile_region(state, range.first, range.second, true);

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);
	}

	void set_tile_region_node(ExecutionState *state, unsigned left, unsigned right) const
	{
		node_context *context = reinterpret_cast<node_context *>(state->get_context(m_id));
		context->assert_guard_pattern();

		auto range = m_filter->get_required_col_range(left, right);

		m_data.node_info.parent->set_tile_region(state, range.first, range.second, false);
		if (m_data.node_info.parent_uv)
			m_data.node_info.parent_uv->set_tile_region(state, range.first, range.second, true);

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);
	}

	const ZimgImageBufferConst *generate_line_source(ExecutionState *state, unsigned i, bool uv)
	{
		node_context *context = reinterpret_cast<node_context *>(state->get_context(m_id));
		context->assert_guard_pattern();

		unsigned step = 1 << m_data.source_info.subsample_h;
		unsigned line = uv ? i * step : i;
		unsigned pos = context->cache_pos;

		if (line >= pos) {
			if (state->get_unpack_cb()) {
				for (; pos <= line; pos += step) {
					state->get_unpack_cb()(pos, context->source_left, context->source_right);
				}
			} else {
				pos = mod(line, step) + step;
			}

			context->cache_pos = pos;
		}

		return &state->get_input_buffer();
	}

	const ZimgImageBufferConst *generate_line_node_uv(ExecutionState *state, const ZimgImageBuffer *external, unsigned i)
	{
		node_context *context = reinterpret_cast<node_context *>(state->get_context(m_id));
		context->assert_guard_pattern();

		const ZimgImageBuffer *output_buffer = external ? external : &context->cache_buf;
		unsigned pos = context->cache_pos;

		for (; pos <= i; pos += m_data.node_info.step) {
			const ZimgImageBufferConst *input_buffer = nullptr;
			ZimgImageBufferConst input_buffer_one;
			ZimgImageBuffer output_buffer_one;

			auto range = m_filter->get_required_row_range(pos);

			for (unsigned ii = range.first; ii < range.second; ++ii) {
				input_buffer = m_data.node_info.parent->generate_line(state, nullptr, ii, true);
			}

			for (unsigned p = 1; p < 3; ++p) {
				void *filter_ctx = p == 1 ? context->filter_ctx : context->filter_ctx2;

				input_buffer_one.data[0] = input_buffer->data[p];
				input_buffer_one.stride[0] = input_buffer->stride[p];
				input_buffer_one.mask[0] = input_buffer->mask[p];

				output_buffer_one.data[0] = output_buffer->data[p];
				output_buffer_one.stride[0] = output_buffer->stride[p];
				output_buffer_one.mask[0] = output_buffer->mask[p];

				m_filter->process(filter_ctx, input_buffer_one, output_buffer_one, state->get_tmp(), pos, context->source_left, context->source_right);
			}
		}
		context->cache_pos = pos;

		return &static_cast<const ZimgImageBufferConst &>(*output_buffer);
	}

	const ZimgImageBufferConst *generate_line_node(ExecutionState *state, const ZimgImageBuffer *external, unsigned i)
	{
		node_context *context = reinterpret_cast<node_context *>(state->get_context(m_id));
		context->assert_guard_pattern();

		const ZimgImageBuffer *output_buffer = external ? external : &context->cache_buf;
		unsigned pos = context->cache_pos;

		for (; pos <= i; pos += m_data.node_info.step) {
			const ZimgImageBufferConst *input_buffer = nullptr;
			const ZimgImageBufferConst *input_buffer_uv = nullptr;

			auto range = m_filter->get_required_row_range(pos);

			for (unsigned ii = range.first; ii < range.second; ++ii) {
				input_buffer = m_data.node_info.parent->generate_line(state, nullptr, ii, false);

				if (m_data.node_info.parent_uv)
					input_buffer_uv = m_data.node_info.parent_uv->generate_line(state, nullptr, ii, true);
			}

			if (m_data.node_info.parent_uv) {
				ZimgImageBufferConst input_buffer_yuv;

				input_buffer_yuv.data[0] = input_buffer->data[0];
				input_buffer_yuv.stride[0] = input_buffer->stride[0];
				input_buffer_yuv.mask[0] = input_buffer->mask[0];

				for (unsigned p = 1; p < 3; ++p) {
					input_buffer_yuv.data[p] = input_buffer_uv->data[p];
					input_buffer_yuv.stride[p] = input_buffer_uv->stride[p];
					input_buffer_yuv.mask[p] = input_buffer_uv->mask[p];
				}

				m_filter->process(context->filter_ctx, input_buffer_yuv, *output_buffer, state->get_tmp(), pos, context->source_left, context->source_right);
			} else {
				m_filter->process(context->filter_ctx, *input_buffer, *output_buffer, state->get_tmp(), pos, context->source_left, context->source_right);
			}
		}
		context->cache_pos = pos;

		return &static_cast<const ZimgImageBufferConst &>(*output_buffer);
	}
public:
	GraphNode(source_tag tag, unsigned id, unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color) :
		m_data{ tag },
		m_cache_lines{ 1U << subsample_h },
		m_ref_count{},
		m_id{ id },
		m_is_source{ true }
	{
		m_data.source_info.width = width;
		m_data.source_info.height = height;
		m_data.source_info.type = type;
		m_data.source_info.subsample_w = subsample_w;
		m_data.source_info.subsample_h = subsample_h;
		m_data.source_info.color = color;
	}

	GraphNode(filter_tag tag, unsigned id, GraphNode *parent, GraphNode *parent_uv, IZimgFilter *filter) :
		m_data{ tag },
		m_cache_lines{},
		m_ref_count{},
		m_id{ id },
		m_is_source{ false }
	{
		m_data.node_info.parent = parent;
		m_data.node_info.parent_uv = parent_uv;
		m_data.node_info.flags = filter->get_flags();
		m_data.node_info.step = filter->get_simultaneous_lines();
		m_data.node_info.is_uv = false;

		m_filter.reset(filter);
	}

	GraphNode(filter_uv_tag tag, unsigned id, GraphNode *parent, IZimgFilter *filter) :
		m_data{ tag },
		m_cache_lines{},
		m_ref_count{},
		m_id{ id },
		m_is_source{ false }
	{
		m_data.node_info.parent = parent;
		m_data.node_info.parent_uv = nullptr;
		m_data.node_info.flags = filter->get_flags();
		m_data.node_info.step = filter->get_simultaneous_lines();
		m_data.node_info.is_uv = true;

		m_filter.reset(filter);
	}

	unsigned add_ref()
	{
		return ++m_ref_count;
	}

	unsigned get_ref() const
	{
		return m_ref_count;
	}

	void simulate(SimulationState *sim, unsigned first, unsigned last, bool uv = false)
	{
		if (m_is_source)
			simulate_source(sim, first, last, uv);
		else if (m_data.node_info.is_uv)
			simulate_node_uv(sim, first, last);
		else
			simulate_node(sim, first, last);
	}

	bool entire_row() const
	{
		bool entire_row = false;

		if (!m_is_source) {
			entire_row = entire_row || m_data.node_info.flags.entire_row;
			entire_row = entire_row || m_data.node_info.parent->entire_row();

			if (m_data.node_info.parent_uv)
				entire_row = entire_row || m_data.node_info.parent_uv->entire_row();
		}

		return entire_row;
	}

	IZimgFilter::image_attributes get_image_attributes(bool uv = false) const
	{
		if (m_is_source) {
			unsigned width = m_data.source_info.width >> (uv ? m_data.source_info.subsample_w : 0);
			unsigned height = m_data.source_info.height >> (uv ? m_data.source_info.subsample_h : 0);
			return{ width, height, m_data.source_info.type };
		} else {
			return m_filter->get_image_attributes();
		}
	}

	size_t get_context_size() const
	{
		FakeAllocator alloc;

		alloc.allocate_n<node_context>(1);

		if (!m_is_source) {
			unsigned num_planes = get_num_planes();
			unsigned cache_lines = get_real_cache_lines();
			ptrdiff_t stride = get_cache_stride();

			alloc.allocate((size_t)num_planes * cache_lines * stride);
			alloc.allocate(m_filter->get_context_size());

			if (m_data.node_info.is_uv)
				alloc.allocate(m_filter->get_context_size());
		}

		return alloc.count();
	}

	size_t get_tmp_size(unsigned left, unsigned right) const
	{
		size_t tmp_size = 0;

		if (!m_is_source) {
			auto range = m_filter->get_required_col_range(left, right);

			tmp_size = std::max(tmp_size, m_filter->get_tmp_size(left, right));
			tmp_size = std::max(tmp_size, m_data.node_info.parent->get_tmp_size(range.first, range.second));

			if (m_data.node_info.parent_uv)
				tmp_size = std::max(tmp_size, m_data.node_info.parent_uv->get_tmp_size(range.first, range.second));
		}

		return tmp_size;
	}

	unsigned get_cache_lines() const
	{
		return m_cache_lines;
	}

	void init_context(ExecutionState *state)
	{
		size_t context_size = get_context_size();
		LinearAllocator alloc{ state->alloc_context(m_id, context_size) };

		node_context *context = new (alloc.allocate_n<node_context>(1)) node_context{};
		context->cache_pos = 0;

		if (!m_is_source && m_data.node_info.is_uv)
			init_context_node_uv(alloc, context);
		else if (!m_is_source)
			init_context_node(alloc, context);

		_zassert(alloc.count() <= context_size, "buffer overflow detected");
		_zassert_d(alloc.count() == context_size, "allocation mismatch");
	}

	void reset_context(ExecutionState *state)
	{
		auto attr = get_image_attributes();

		node_context *context = reinterpret_cast<node_context *>(state->get_context(m_id));
		context->assert_guard_pattern();
		context->cache_pos = 0;
		context->source_left = attr.width;
		context->source_right = 0;

		if (!m_is_source && m_data.node_info.is_uv) {
			m_filter->init_context(context->filter_ctx);
			m_filter->init_context(context->filter_ctx2);
		} else if (!m_is_source) {
			m_filter->init_context(context->filter_ctx);
		}
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv)
	{
		if (m_is_source)
			set_tile_region_source(state, left, right, uv);
		else if (m_data.node_info.is_uv)
			set_tile_region_node_uv(state, left, right);
		else
			set_tile_region_node(state, left, right);
	}

	const ZimgImageBufferConst *generate_line(ExecutionState *state, const ZimgImageBuffer *external, unsigned i, bool uv)
	{
		if (m_is_source)
			return generate_line_source(state, i, uv);
		else if (m_data.node_info.is_uv)
			return generate_line_node_uv(state, external, i);
		else
			return generate_line_node(state, external, i);
	}
};

const GraphNode::source_tag GraphNode::SOURCE{};
const GraphNode::filter_tag GraphNode::FILTER{};
const GraphNode::filter_uv_tag GraphNode::FILTER_UV{};

} // namespace


class FilterGraph::impl {
	static const unsigned HORIZONTAL_STEP = 512;
	static const unsigned TILE_MIN = 64;

	std::vector<std::unique_ptr<GraphNode>> m_node_set;
	GraphNode *m_head;
	GraphNode *m_node;
	GraphNode *m_node_uv;
	unsigned m_id_counter;
	unsigned m_subsample_w;
	unsigned m_subsample_h;
	bool m_is_complete;

	unsigned get_horizontal_step() const
	{
		auto head_attr = m_head->get_image_attributes();
		auto tail_attr = m_node->get_image_attributes();

		bool entire_row = m_node->entire_row() || (m_node_uv && m_node_uv->entire_row());

		if (!entire_row) {
			double scale = std::max((double)tail_attr.width / head_attr.width, 1.0);
			unsigned step = mod((unsigned)std::lrint(HORIZONTAL_STEP * scale), ALIGNMENT);
			return std::min(step, tail_attr.width);
		} else {
			return tail_attr.width;
		}
	}

	void check_incomplete() const
	{
		if (m_is_complete)
			throw error::InternalError{ "cannot modify completed graph" };
	}

	void check_complete() const
	{
		if (!m_is_complete)
			throw error::InternalError{ "cannot query properties on incomplete graph" };
	}
public:
	impl(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color) :
		m_head{},
		m_node{},
		m_node_uv{},
		m_id_counter{},
		m_subsample_w{},
		m_subsample_h{},
		m_is_complete{}
	{
		if (!color && (subsample_w || subsample_h))
			throw error::InternalError{ "greyscale images can not be subsampled" };
		if (subsample_w > 2 || subsample_h > 2)
			throw error::UnsupportedSubsampling{ "subsampling factor must not exceed 4" };

		m_node_set.emplace_back(new GraphNode{ GraphNode::SOURCE, m_id_counter++, width, height, type, subsample_w, subsample_h, color });
		m_head = m_node_set.back().get();
		m_node = m_head;

		if (color)
			m_node_uv = m_head;
	}

	void attach_filter(IZimgFilter *filter)
	{
		check_incomplete();

		ZimgFilterFlags flags = filter->get_flags();
		GraphNode *parent = m_node;
		GraphNode *parent_uv = nullptr;

		if (flags.color) {
			auto attr = m_node->get_image_attributes();
			auto attr_uv = m_node->get_image_attributes();

			if (!m_node_uv)
				throw error::InternalError{ "cannot use color filter in greyscale graph" };
			if (attr.width != attr_uv.width || attr.height != attr_uv.height || attr.type != attr_uv.type)
				throw error::InternalError{ "cannot use color filter with mismatching Y and UV format" };

			parent_uv = m_node_uv;
		}

		m_node_set.emplace_back(new GraphNode{ GraphNode::FILTER, m_id_counter++, parent, parent_uv, filter });
		m_node = m_node_set.back().get();

		parent->add_ref();
		if (parent_uv)
			parent_uv->add_ref();

		if (flags.color)
			m_node_uv = m_node;
	}

	void attach_filter_uv(IZimgFilter *filter)
	{
		check_incomplete();

		ZimgFilterFlags flags = filter->get_flags();
		GraphNode *parent = m_node_uv;

		if (flags.color)
			throw error::InternalError{ "cannot use color filter as UV filter" };

		m_node_set.emplace_back(new GraphNode{ GraphNode::FILTER_UV, m_id_counter++, parent, filter });
		m_node_uv = m_node_set.back().get();
		parent->add_ref();
	}

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
			throw error::InternalError{ "unsupported horizontal subsampling" };
		if (node_attr.height != node_attr_uv.height << subsample_h)
			throw error::InternalError{ "unsupported vertical subsampling" };
		if (node_attr.type != node_attr_uv.type)
			throw error::InternalError{ "UV pixel type can not differ" };

		if (m_node == m_head || m_node->get_ref())
			attach_filter(new CopyFilter{ node_attr.width, node_attr.height, node_attr.type });
		if (m_node_uv && (m_node_uv == m_head || m_node_uv->get_ref()))
			attach_filter_uv(new CopyFilter{ node_attr_uv.width, node_attr_uv.height, node_attr_uv.type });

		SimulationState sim{ m_id_counter };

		for (unsigned i = 0; i < node_attr.height; i += (1 << subsample_h)) {
			m_node->simulate(&sim, i, i + (1 << subsample_h));

			if (m_node_uv)
				m_node_uv->simulate(&sim, i >> subsample_h, (i >> subsample_h) + 1, true);
		}

		m_subsample_w = subsample_w;
		m_subsample_h = subsample_h;
		m_is_complete = true;
	}

	size_t get_tmp_size() const
	{
		check_complete();

		auto attr = m_node->get_image_attributes();
		unsigned step = get_horizontal_step();

		FakeAllocator alloc;
		size_t tmp_size = 0;

		alloc.allocate(ExecutionState::context_table_size(m_id_counter));

		for (const auto &node : m_node_set) {
			alloc.allocate(node->get_context_size());
		}

		for (unsigned j = 0; j < attr.width; j += step) {
			unsigned j_end = std::min(j + step, attr.width);

			tmp_size = std::max(tmp_size, m_node->get_tmp_size(j, j_end));

			if (m_node_uv)
				tmp_size = std::max(tmp_size, m_node_uv->get_tmp_size(j >> m_subsample_w, j_end >> m_subsample_w));
		}
		alloc.allocate(tmp_size);

		return alloc.count();
	}

	unsigned get_input_buffering() const
	{
		check_complete();
		return m_head->get_cache_lines();
	}

	unsigned get_output_buffering() const
	{
		check_complete();

		unsigned lines = m_node->get_cache_lines();

		if (m_node_uv) {
			unsigned lines_uv = m_node_uv->get_cache_lines();
			lines_uv = lines_uv == (unsigned)-1 ? lines_uv : lines_uv << m_subsample_h;
			lines = std::max(lines, lines_uv);
		}

		return lines;
	}

	void process(const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *tmp, callback unpack_cb, callback pack_cb) const
	{
		check_complete();

		ExecutionState state{ m_id_counter, src, dst, tmp, unpack_cb, pack_cb };
		auto attr = m_node->get_image_attributes();
		unsigned h_step = get_horizontal_step();
		unsigned v_step = 1 << m_subsample_h;

		for (const auto &node : m_node_set) {
			node->init_context(&state);
		}

		for (unsigned j = 0; j < attr.width; j += h_step) {
			unsigned j_end = std::min(j + h_step, attr.width);

			if (attr.width - j_end < TILE_MIN) {
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
					m_node->generate_line(&state, &dst, ii, false);
				}

				if (m_node_uv)
					m_node_uv->generate_line(&state, &state.get_output_buffer(), i / v_step, true);

				if (state.get_pack_cb())
					state.get_pack_cb()(i, j, j_end);
			}
		}

	}
};


FilterGraph::callback::callback(std::nullptr_t) :
	m_func{},
	m_user{}
{
}

FilterGraph::callback::callback(func_type func, void *user) :
	m_func{ func },
	m_user{ user }
{
}

FilterGraph::callback::operator bool() const
{
	return m_func != nullptr;
}

void FilterGraph::callback::operator()(unsigned i, unsigned left, unsigned right) const
{
	if (m_func(m_user, i, left, right))
		throw error::UserCallbackFailed{ "user callback failed" };
}


FilterGraph::FilterGraph(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color) :
	m_impl{ new impl{ width, height, type, subsample_w, subsample_h, color } }
{
}

FilterGraph::~FilterGraph()
{
}

void FilterGraph::attach_filter(IZimgFilter *filter)
{
	m_impl->attach_filter(filter);
}

void FilterGraph::attach_filter_uv(IZimgFilter *filter)
{
	m_impl->attach_filter_uv(filter);
}

void FilterGraph::complete()
{
	m_impl->complete();
}

size_t FilterGraph::get_tmp_size() const
{
	return m_impl->get_tmp_size();
}

unsigned FilterGraph::get_input_buffering() const
{
	return m_impl->get_input_buffering();
}

unsigned FilterGraph::get_output_buffering() const
{
	return m_impl->get_output_buffering();
}

void FilterGraph::process(const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *tmp, callback unpack_cb, callback pack_cb) const
{
	m_impl->process(src, dst, tmp, unpack_cb, pack_cb);
}

} // namespace zimg
