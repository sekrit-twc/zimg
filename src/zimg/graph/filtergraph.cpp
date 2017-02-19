#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "common/align.h"
#include "common/alloc.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "basic_filter.h"
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
	{
	}

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
			m_value.w = static_cast<uint8_t>(1U << (depth - 1));
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
	const ImageBuffer<const void> *m_src_buf;
	const ImageBuffer<void> *m_dst_buf;
	FilterGraph::callback m_unpack_cb;
	FilterGraph::callback m_pack_cb;
	void **m_context_table;
	void *m_base;
public:
	ExecutionState(unsigned id_counter, const ImageBuffer<const void> src_buf[], const ImageBuffer<void> dst_buf[], void *pool,
	               FilterGraph::callback unpack_cb, FilterGraph::callback pack_cb) :
		m_alloc{ pool },
		m_src_buf{ src_buf },
		m_dst_buf{ dst_buf },
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
		zassert(!m_context_table[id], "context already allocated");

		m_context_table[id] = m_alloc.allocate(size);
		return m_context_table[id];
	}

	void *get_context(unsigned id) const
	{
		return m_context_table[id];
	}

	const ImageBuffer<const void> *get_input_buffer() const
	{
		return m_src_buf;
	}

	const ImageBuffer<void> *get_output_buffer() const
	{
		return m_dst_buf;
	}

	void *get_tmp() const
	{
		return static_cast<char *>(m_base) + m_alloc.count();
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
		return ceil_n(sizeof(void *) * id_counter, ALIGNMENT);
	}
};


class GraphNode {
protected:
	struct node_context {
		static constexpr uint64_t GUARD_PATTERN = 0xDEADBEEFDEADBEEFULL;
		const uint64_t guard_pattern = GUARD_PATTERN;
		unsigned cache_pos;
		unsigned source_left;
		unsigned source_right;

		void assert_guard_pattern() const
		{
			zassert_d(guard_pattern == GUARD_PATTERN, "buffer overflow detected");
		}

	};
private:
	unsigned m_id;
	unsigned m_ref_count;
	unsigned m_cache_lines;
protected:
	GraphNode(unsigned id) :
		m_id{ id },
		m_ref_count{},
		m_cache_lines{}
	{
	}

	unsigned get_id() const
	{
		return m_id;
	}

	unsigned get_real_cache_lines() const
	{
		return get_cache_lines() == BUFFER_MAX ? get_image_attributes(false).height : get_cache_lines();
	}

	ptrdiff_t get_cache_stride() const
	{
		auto attr = get_image_attributes(false);
		return ceil_n(attr.width * pixel_size(attr.type), ALIGNMENT);
	}

	void set_cache_lines(unsigned n)
	{
		if (n > m_cache_lines) {
			if (n >= get_image_attributes().height)
				m_cache_lines = BUFFER_MAX;
			else
				m_cache_lines = select_zimg_buffer_mask(n) + 1;
		}
	}

	void init_context_base(node_context *ctx) const
	{
		ctx->cache_pos = 0;
		ctx->source_left = 0;
		ctx->source_right = 0;
	}

	void reset_context_base(node_context *ctx) const
	{
		auto attr = get_image_attributes(false);

		ctx->cache_pos = 0;
		ctx->source_left = attr.width;
		ctx->source_right = 0;
	}
public:
	virtual ~GraphNode() = default;

	unsigned add_ref()
	{
		return ++m_ref_count;
	}

	unsigned get_ref() const
	{
		return m_ref_count;
	}

	unsigned get_cache_lines() const
	{
		return m_cache_lines;
	}

	virtual ImageFilter::image_attributes get_image_attributes(bool uv = false) const = 0;

	virtual bool entire_row() const = 0;

	virtual void simulate(SimulationState *sim, unsigned first, unsigned last, bool uv = false) = 0;

	virtual size_t get_context_size() const = 0;

	virtual size_t get_tmp_size(unsigned left, unsigned right) const = 0;

	virtual void init_context(ExecutionState *state) const = 0;

	virtual void reset_context(ExecutionState *state) const = 0;

	virtual void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const = 0;

	virtual const ImageBuffer<const void> *generate_line(ExecutionState *state, const ImageBuffer<void> *external, unsigned i, bool uv) const = 0;
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
	{
	}

	ImageFilter::image_attributes get_image_attributes(bool uv) const override
	{
		auto attr = m_attr;

		if (uv) {
			attr.width >>= m_subsample_w;
			attr.height >>= m_subsample_h;
		}
		return attr;
	}

	bool entire_row() const override
	{
		return false;
	}

	void simulate(SimulationState *sim, unsigned first, unsigned last, bool uv) override
	{
		unsigned step = 1 << m_subsample_h;
		unsigned pos = sim->pos(get_id());

		first <<= uv ? m_subsample_h : 0;
		last <<= uv ? m_subsample_h : 0;

		if (pos < last)
			pos = floor_n(last - 1, step) + step;

		sim->pos(get_id()) = pos;
		set_cache_lines(pos - first);
	}

	size_t get_context_size() const override
	{
		return ceil_n(sizeof(node_context), zimg::ALIGNMENT);
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		return 0;
	}

	void init_context(ExecutionState *state) const override
	{
		size_t context_size = get_context_size();
		LinearAllocator alloc{ state->alloc_context(get_id(), get_context_size()) };

		node_context *context = new (alloc.allocate_n<node_context>(1)) node_context{};
		init_context_base(context);

		zassert(alloc.count() <= context_size, "buffer overflow detected");
		zassert_d(alloc.count() == context_size, "allocation mismatch");
	}

	void reset_context(ExecutionState *state) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		reset_context_base(context);
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		context->assert_guard_pattern();

		if (uv) {
			left <<= m_subsample_w;
			right <<= m_subsample_w;
		}

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);
	}

	const ImageBuffer<const void> *generate_line(ExecutionState *state, const ImageBuffer<void> *external, unsigned i, bool uv) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		context->assert_guard_pattern();

		unsigned step = 1 << m_subsample_h;
		unsigned line = uv ? i * step : i;
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

		return static_buffer_cast<const void>(state->get_input_buffer());
	}
};

class FilterNode final : public GraphNode {
	struct node_context : public GraphNode::node_context {
		void *filter_ctx;
		ImageBuffer<void> cache_buf[3];
	};

	std::unique_ptr<ImageFilter> m_filter;
	ImageFilter::filter_flags m_flags;
	GraphNode *m_parent;
	GraphNode *m_parent_uv;
	unsigned m_step;

	unsigned get_num_planes() const
	{
		return m_flags.color ? 3U : 1U;
	}
public:
	FilterNode(unsigned id, std::unique_ptr<ImageFilter> &&filter, GraphNode *parent, GraphNode *parent_uv) :
		GraphNode(id),
		m_flags(filter->get_flags()),
		m_parent{ parent },
		m_parent_uv{ parent_uv },
		m_step{ filter->get_simultaneous_lines() }
	{
		m_filter = std::move(filter);
	}

	ImageFilter::image_attributes get_image_attributes(bool) const override
	{
		return m_filter->get_image_attributes();
	}

	bool entire_row() const override
	{
		return m_flags.entire_row || m_parent->entire_row() || (m_parent_uv && m_parent_uv->entire_row());
	}

	void simulate(SimulationState *sim, unsigned first, unsigned last, bool) override
	{
		unsigned pos = sim->pos(get_id());

		for (; pos < last; pos += m_step) {
			auto range = m_filter->get_required_row_range(pos);

			m_parent->simulate(sim, range.first, range.second, false);

			if (m_parent_uv)
				m_parent_uv->simulate(sim, range.first, range.second, true);
		}

		sim->pos(get_id()) = pos;
		set_cache_lines(pos - first);
	}

	size_t get_context_size() const override
	{
		FakeAllocator alloc;

		alloc.allocate_n<node_context>(1);

		unsigned num_planes = get_num_planes();
		unsigned cache_lines = get_real_cache_lines();
		ptrdiff_t stride = get_cache_stride();

		if (cache_lines > static_cast<size_t>(PTRDIFF_MAX / stride))
			throw error::OutOfMemory{};

		alloc.allocate(static_cast<checked_size_t>(num_planes) * cache_lines * stride);
		alloc.allocate(m_filter->get_context_size());

		return alloc.count();
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		size_t tmp_size = 0;

		auto range = m_filter->get_required_col_range(left, right);

		tmp_size = std::max(tmp_size, m_filter->get_tmp_size(left, right));
		tmp_size = std::max(tmp_size, m_parent->get_tmp_size(range.first, range.second));

		if (m_parent_uv)
			tmp_size = std::max(tmp_size, m_parent_uv->get_tmp_size(range.first, range.second));

		return tmp_size;
	}

	void init_context(ExecutionState *state) const override
	{
		size_t context_size = get_context_size();
		LinearAllocator alloc{ state->alloc_context(get_id(), context_size) };

		node_context *context = new (alloc.allocate_n<node_context>(1)) node_context{};
		init_context_base(context);

		ptrdiff_t stride = get_cache_stride();
		unsigned cache_lines = get_real_cache_lines();
		unsigned mask = select_zimg_buffer_mask(get_cache_lines());

		context->filter_ctx = alloc.allocate(m_filter->get_context_size());

		for (unsigned p = 0; p < get_num_planes(); ++p) {
			context->cache_buf[p] = { alloc.allocate(static_cast<size_t>(cache_lines) * stride), stride, mask };
		}

		zassert(alloc.count() <= context_size, "buffer overflow detected");
		zassert_d(alloc.count() == context_size, "allocation mismatch");
	}

	void reset_context(ExecutionState *state) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		reset_context_base(context);

		m_filter->init_context(context->filter_ctx);
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		context->assert_guard_pattern();

		auto range = m_filter->get_required_col_range(left, right);

		m_parent->set_tile_region(state, range.first, range.second, false);
		if (m_parent_uv)
			m_parent_uv->set_tile_region(state, range.first, range.second, true);

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);
	}

	const ImageBuffer<const void> *generate_line(ExecutionState *state, const ImageBuffer<void> *external, unsigned i, bool uv) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		context->assert_guard_pattern();

		const ImageBuffer<void> *output_buffer = external ? external : context->cache_buf;
		unsigned pos = context->cache_pos;

		for (; pos <= i; pos += m_step) {
			const ImageBuffer<const void> *input_buffer = nullptr;
			const ImageBuffer<const void> *input_buffer_uv = nullptr;

			auto range = m_filter->get_required_row_range(pos);
			zassert_d(range.first < range.second, "bad row range");

			for (unsigned ii = range.first; ii < range.second; ++ii) {
				input_buffer = m_parent->generate_line(state, nullptr, ii, false);

				if (m_parent_uv)
					input_buffer_uv = m_parent_uv->generate_line(state, nullptr, ii, true);
			}

			if (m_parent_uv) {
				ImageBuffer<const void> input_buffer_yuv[3] = { input_buffer[0], input_buffer_uv[1], input_buffer_uv[2] };

				m_filter->process(context->filter_ctx, input_buffer_yuv, output_buffer, state->get_tmp(), pos, context->source_left, context->source_right);
			} else {
				m_filter->process(context->filter_ctx, input_buffer, output_buffer, state->get_tmp(), pos, context->source_left, context->source_right);
			}
		}
		context->cache_pos = pos;

		return static_buffer_cast<const void>(output_buffer);
	}
};


class FilterNodeUV final : public GraphNode {
	struct node_context : public GraphNode::node_context {
		void *filter_ctx_u;
		void *filter_ctx_v;
		ImageBuffer<void> cache_buf[3];
	};

	std::unique_ptr<ImageFilter> m_filter;
	ImageFilter::filter_flags m_flags;
	GraphNode *m_parent;
	unsigned m_step;

	unsigned get_num_planes() const
	{
		return 2;
	}
public:
	FilterNodeUV(unsigned id, std::unique_ptr<ImageFilter> &&filter, GraphNode *parent) :
		GraphNode(id),
		m_flags(filter->get_flags()),
		m_parent{ parent },
		m_step{ filter->get_simultaneous_lines() }
	{
		m_filter = std::move(filter);
	}

	ImageFilter::image_attributes get_image_attributes(bool) const override
	{
		return m_filter->get_image_attributes();
	}

	bool entire_row() const override
	{
		return m_flags.entire_row || m_parent->entire_row();
	}

	void simulate(SimulationState *sim, unsigned first, unsigned last, bool) override
	{
		unsigned pos = sim->pos(get_id());

		for (; pos < last; pos += m_step) {
			auto range = m_filter->get_required_row_range(pos);
			m_parent->simulate(sim, range.first, range.second, true);
		}

		sim->pos(get_id()) = pos;
		set_cache_lines(pos - first);
	}

	size_t get_context_size() const override
	{
		FakeAllocator alloc;

		alloc.allocate_n<node_context>(1);

		unsigned num_planes = get_num_planes();
		unsigned cache_lines = get_real_cache_lines();
		ptrdiff_t stride = get_cache_stride();

		if (cache_lines > static_cast<size_t>(PTRDIFF_MAX / stride))
			throw error::OutOfMemory{};

		alloc.allocate(static_cast<checked_size_t>(num_planes) * cache_lines * stride);
		alloc.allocate(m_filter->get_context_size());
		alloc.allocate(m_filter->get_context_size());

		return alloc.count();
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		size_t tmp_size = 0;

		auto range = m_filter->get_required_col_range(left, right);

		tmp_size = std::max(tmp_size, m_filter->get_tmp_size(left, right));
		tmp_size = std::max(tmp_size, m_parent->get_tmp_size(range.first, range.second));

		return tmp_size;
	}

	void init_context(ExecutionState *state) const override
	{
		size_t context_size = get_context_size();
		LinearAllocator alloc{ state->alloc_context(get_id(), context_size) };

		node_context *context = new (alloc.allocate_n<node_context>(1)) node_context{};
		init_context_base(context);

		ptrdiff_t stride = get_cache_stride();
		unsigned cache_lines = get_real_cache_lines();
		unsigned mask = select_zimg_buffer_mask(get_cache_lines());

		context->filter_ctx_u = alloc.allocate(m_filter->get_context_size());
		context->filter_ctx_v = alloc.allocate(m_filter->get_context_size());

		context->cache_buf[1] = { alloc.allocate(static_cast<size_t>(cache_lines) * stride), stride, mask };
		context->cache_buf[2] = { alloc.allocate(static_cast<size_t>(cache_lines) * stride), stride, mask };

		zassert(alloc.count() <= context_size, "buffer overflow detected");
		zassert_d(alloc.count() == context_size, "allocation mismatch");
	}

	void reset_context(ExecutionState *state) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		reset_context_base(context);

		m_filter->init_context(context->filter_ctx_u);
		m_filter->init_context(context->filter_ctx_v);
	}

	void set_tile_region(ExecutionState *state, unsigned left, unsigned right, bool uv) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		context->assert_guard_pattern();

		auto range = m_filter->get_required_col_range(left, right);

		m_parent->set_tile_region(state, range.first, range.second, true);

		context->source_left = std::min(context->source_left, left);
		context->source_right = std::max(context->source_right, right);
	}

	const ImageBuffer<const void> *generate_line(ExecutionState *state, const ImageBuffer<void> *external, unsigned i, bool uv) const override
	{
		node_context *context = static_cast<node_context *>(state->get_context(get_id()));
		context->assert_guard_pattern();

		const ImageBuffer<void> *output_buffer = external ? external : context->cache_buf;
		unsigned pos = context->cache_pos;

		for (; pos <= i; pos += m_step) {
			const ImageBuffer<const void> *input_buffer = nullptr;

			auto range = m_filter->get_required_row_range(pos);
			zassert_d(range.first < range.second, "bad row range");

			for (unsigned ii = range.first; ii < range.second; ++ii) {
				input_buffer = m_parent->generate_line(state, nullptr, ii, true);
			}

			m_filter->process(context->filter_ctx_u, input_buffer + 1, output_buffer + 1, state->get_tmp(), pos, context->source_left, context->source_right);
			m_filter->process(context->filter_ctx_v, input_buffer + 2, output_buffer + 2, state->get_tmp(), pos, context->source_left, context->source_right);
		}
		context->cache_pos = pos;

		return static_buffer_cast<const void>(output_buffer);
	}
};

} // namespace


class FilterGraph::impl {
	static constexpr unsigned HORIZONTAL_STEP = 512;
	static constexpr unsigned TILE_MIN = 64;

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
			double scale = std::max(static_cast<double>(tail_attr.width) / head_attr.width, 1.0);
			double step = std::min(HORIZONTAL_STEP * scale, static_cast<double>(tail_attr.width));
			return floor_n(static_cast<unsigned>(std::lrint(step)), ALIGNMENT);
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
		zassert_d(width <= pixel_max_width(type), "image stride causes overflow");

		if (!color && (subsample_w || subsample_h))
			throw error::InternalError{ "greyscale images can not be subsampled" };
		if (subsample_w > 2 || subsample_h > 2)
			throw error::UnsupportedSubsampling{ "subsampling factor must not exceed 4" };

		m_node_set.emplace_back(
			ztd::make_unique<SourceNode>(m_id_counter++, width, height, type, subsample_w, subsample_h));
		m_head = m_node_set.back().get();
		m_node = m_head;

		if (color)
			m_node_uv = m_head;
	}

	void attach_filter(std::unique_ptr<ImageFilter> &&filter)
	{
		check_incomplete();

		ImageFilter::filter_flags flags = filter->get_flags();
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

		m_node_set.reserve(m_node_set.size() + 1);
		m_node_set.emplace_back(
			ztd::make_unique<FilterNode>(m_id_counter++, std::move(filter), parent, parent_uv));
		m_node = m_node_set.back().get();

		parent->add_ref();
		if (parent_uv)
			parent_uv->add_ref();

		if (flags.color)
			m_node_uv = m_node;
	}

	void attach_filter_uv(std::unique_ptr<ImageFilter> &&filter)
	{
		check_incomplete();

		if (filter->get_flags().color)
			throw error::InternalError{ "cannot use color filter as UV filter" };

		GraphNode *parent = m_node_uv;

		m_node_set.reserve(m_node_set.size() + 1);
		m_node_set.emplace_back(
			ztd::make_unique<FilterNodeUV>(m_id_counter++, std::move(filter), parent));
		m_node_uv = m_node_set.back().get();
		parent->add_ref();
	}

	void color_to_grey()
	{
		check_incomplete();

		if (!m_node_uv)
			throw error::InternalError{ "cannot remove chroma from greyscale image" };

		ImageFilter::image_attributes attr = m_node->get_image_attributes();
		GraphNode *parent = m_node;

		m_node_set.reserve(m_node_set.size() + 1);
		m_node_set.emplace_back(
			ztd::make_unique<FilterNode>(m_id_counter++, ztd::make_unique<CopyFilter>(attr.width, attr.height, attr.type), parent, nullptr));
		m_node = m_node_set.back().get();
		m_node_uv = nullptr;

		parent->add_ref();
	}

	void grey_to_color(bool yuv, unsigned subsample_w, unsigned subsample_h, unsigned depth)
	{
		check_incomplete();

		if (m_node_uv)
			throw error::InternalError{ "cannot add chroma to color image" };

		ImageFilter::image_attributes attr = m_node->get_image_attributes();
		GraphNode *parent = m_node;

		m_node_set.emplace_back(
			ztd::make_unique<FilterNode>(m_id_counter++, ztd::make_unique<ColorExtendFilter>(attr, !yuv), parent, nullptr));
		m_node = m_node_set.back().get();
		m_node_uv = m_node;

		parent->add_ref();

		if (yuv)
			attach_filter_uv(ztd::make_unique<ChromaInitializeFilter>(attr, subsample_w, subsample_h, depth));
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
			attach_filter(ztd::make_unique<CopyFilter>(node_attr.width, node_attr.height, node_attr.type));
		if (m_node_uv && (m_node_uv == m_head || m_node_uv->get_ref()))
			attach_filter_uv(ztd::make_unique<CopyFilter>(node_attr_uv.width, node_attr_uv.height, node_attr_uv.type));

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

			if (attr.width - j_end < TILE_MIN) {
				j_end = attr.width;
				step = attr.width - j;
			}

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
			lines_uv = (lines_uv == BUFFER_MAX) ? lines_uv : lines_uv << m_subsample_h;
			lines = std::max(lines, lines_uv);
		}

		return lines;
	}

	void process(const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, callback unpack_cb, callback pack_cb) const
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
					m_node->generate_line(&state, state.get_output_buffer(), ii, false);
				}

				if (m_node_uv)
					m_node_uv->generate_line(&state, state.get_output_buffer(), i / v_step, true);

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
	int ret;

	try {
		ret = m_func(m_user, i, left, right);
	} catch (...) {
		zassert_d(false, "user callback must not throw");
		ret = 1;
	}

	if (ret)
		throw error::UserCallbackFailed{ "user callback failed" };
}


FilterGraph::FilterGraph(unsigned width, unsigned height, PixelType type, unsigned subsample_w, unsigned subsample_h, bool color) :
	m_impl{ ztd::make_unique<impl>(width, height, type, subsample_w, subsample_h, color) }
{
}

FilterGraph::FilterGraph(FilterGraph &&other) noexcept = default;

FilterGraph::~FilterGraph() = default;

FilterGraph &FilterGraph::operator=(FilterGraph &&other) noexcept = default;

void FilterGraph::attach_filter(std::unique_ptr<ImageFilter> &&filter)
{
	get_impl()->attach_filter(std::move(filter));
}

void FilterGraph::attach_filter_uv(std::unique_ptr<ImageFilter> &&filter)
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

void FilterGraph::process(const ImageBuffer<const void> *src, const ImageBuffer<void> *dst, void *tmp, callback unpack_cb, callback pack_cb) const
{
	get_impl()->process(src, dst, tmp, unpack_cb, pack_cb);
}

} // namespace graph
} // namespace zimg
