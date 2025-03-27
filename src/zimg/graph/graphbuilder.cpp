#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>
#include "colorspace/colorspace.h"
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "depth/depth.h"
#include "graphengine/filter.h"
#include "graphengine/graph.h"
#include "resize/filter.h"
#include "resize/resize.h"
#include "unresize/unresize.h"
#include "filtergraph.h"
#include "graphbuilder.h"
#include "graphengine_except.h"
#include "simple_filters.h"


#ifndef ZIMG_UNSAFE_IMAGE_SIZE
  #include <limits>
  static constexpr size_t IMAGE_DIMENSION_MAX = static_cast<size_t>(1U) << (std::numeric_limits<size_t>::digits / 2 - 2);
#else
  static constexpr size_t IMAGE_DIMENSION_MAX = ~static_cast<size_t>(0);
#endif // ZIMG_UNSAFE_IMAGE_SIZE


#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)
#define iassert(cond) do { \
  if (!(cond)) \
    error::throw_<error::InternalError>("invalid graph state L" STRINGIFY(__LINE__) ": " #cond); \
  } while (0)


namespace zimg::graph {

namespace {

constexpr int PLANE_NUM = 4;
constexpr int PLANE_Y = 0;
constexpr int PLANE_U = 1;
constexpr int PLANE_V = 2;
constexpr int PLANE_A = 3;
static_assert(PLANE_NUM <= graphengine::NODE_MAX_PLANES);

typedef std::array<bool, PLANE_NUM> plane_mask;

constexpr plane_mask luma_planes{ true, false, false, false };
constexpr plane_mask chroma_planes{ false, true, true, false };
constexpr plane_mask alpha_planes{ false, false, false, true };

constexpr plane_mask operator|(const plane_mask &lhs, const plane_mask &rhs)
{
	return{ lhs[0] || rhs[0], lhs[1] || rhs[1], lhs[2] || rhs[2], lhs[3] || rhs[3] };
}


class DefaultFilterObserver : public FilterObserver {};


// Offset of chroma sample from corresponding centered chroma sample in units of chroma pixels.
double chroma_offset_w(GraphBuilder::ChromaLocationW loc, double subsampling)
{
	// 4:2:0
	// x x x x
	// LCR LCR
	// x x x x
	//
	// L offset = -1/4, R offset = +1/4
	//
	// 4:1:0
	// x x x x x x x x
	// L  C  R L  C  R
	// x x x x x x x x
	//
	// L offset = -3/8, R offset = +3/8
	//
	// x = luma, L = left chroma, C = center chroma, R = right chroma

	if (loc == GraphBuilder::ChromaLocationW::LEFT)
		return -0.5 + 0.5 * subsampling;
	else
		return 0.0;
}

double chroma_offset_h(GraphBuilder::ChromaLocationH loc, double subsampling)
{
	// 4:2:0 (chroma horizontally centered for illustration)
	// xTx
	//  C
	// xBx
	//
	// xTx
	//  C
	// xBx
	//
	// T offset = -1/4, B offset = +1/4
	//
	// x = luma, T = top chroma, C = center chroma, B = bottom chroma

	if (loc == GraphBuilder::ChromaLocationH::TOP)
		return -0.5 + 0.5 * subsampling;
	else if (loc == GraphBuilder::ChromaLocationH::BOTTOM)
		return 0.5 - 0.5 * subsampling;
	else
		return 0.0;
}

// Calculate the offset from a progressive frame with the same dimensions as a field.
double chroma_offset_parity_adjustment(GraphBuilder::FieldParity parity, double offset)
{
	// 4:2:2 (chroma horizontally centered for illustration)
	// INTERLACED PROG_FRAME PROG_FIELD
	//  0 1 2      0 1 2      0 1 2
	// 0          0          0
	//   TtT        xox
	// 1          1          1 xox
	//   BbB        xox
	// 2          2          2
	//   TtT        xox
	// 3          3          3 xox
	//   BbB        xox
	//
	// offset from PROG_FIELD: -1/4 (top field), +1/4 (bottom field)
	//
	// 4:2:0 (horizontally centered for illustration)
	// CTOP       CCENTER    CBOTTOM    PROG_FRAME PROG_FIELD
	//  0 1 2      0 1 2      0 1 2      0 1 2      0 1 2
	// 0          0          0          0          0
	//   TtT        T T        T T        x x
	// 1          1  t       1          1  o       1 x x
	//   B B        B B        BtB        x x
	// 2          2          2          2          2  o
	//   TbT        T T        T T        x x
	// 3          3  b        3         3  o       3 x x
	//   B B        B B        BbB        x x
	// 4          4          4          4          4
	//   TtT        T T        T T        x x
	// 5          5  t        5         5  o       5 x x
	//   B B        B B        BtB        x x
	// 6          6          6          6          6  o
	//   TbT        T T        T T        x x
	// 7          7  b       7          7  o       7 x x
	//   B B        B B        BtB        x x
	//
	// TOP offset from PROG_FIELD: -3/8 (top field), +1/8 (bottom field)
	// CENTER offset from PROG_FIELD: -1/4 (top field), +1/4 (bottom field)
	// BOTTOM offset from PROG_FIELD: -1/8 (top field), +3/8 (bottom field)
	//
	// x = progressive luma, o = progressive chroma
	// T = top field luma, B = bottom field luma
	// t = top field chroma, b = bottomm field chroma

	if (parity == GraphBuilder::FieldParity::TOP)
		return -0.25 + offset / 2;
	else if (parity == GraphBuilder::FieldParity::BOTTOM)
		return 0.25 + offset / 2;
	else
		return offset;
}

double luma_parity_offset(GraphBuilder::FieldParity parity)
{
	// INTERLACED PROG_FRAME PROG_FIELD
	//  0 1 2      0 1 2      0 1 2
	// 0          0          0
	//   T T        x x
	// 1          1          1 x x
	//   B B        x x
	// 2          2          2
	//   T T        x x
	// 3          3          3 x x
	//   B B        x x
	//
	// offset from PROG_FIELD: -1/4 (top field), +1/4 (bottom field)
	//
	// x = progressive, T = top field, B = bottom field

	if (parity == GraphBuilder::FieldParity::TOP)
		return -0.25;
	else if (parity == GraphBuilder::FieldParity::BOTTOM)
		return 0.25;
	else
		return 0.0;
}

void validate_state(const GraphBuilder::state &state)
{
	if (!state.width || !state.height)
		error::throw_<error::InvalidImageSize>("image dimensions must be non-zero");
	if (state.width > IMAGE_DIMENSION_MAX || state.height > IMAGE_DIMENSION_MAX)
		error::throw_<error::InvalidImageSize>("image dimensions exceed implementation limit");
	if (state.width > pixel_max_width(state.type))
		error::throw_<error::InvalidImageSize>("image width exceeds memory addressing limit");

	if (state.color == GraphBuilder::ColorFamily::GREY) {
		if (state.subsample_w || state.subsample_h)
			error::throw_<error::GreyscaleSubsampling>("cannot subsample greyscale image");
		if (state.colorspace.matrix == colorspace::MatrixCoefficients::RGB)
			error::throw_<error::ColorFamilyMismatch>("GREY color family cannot have RGB matrix coefficients");
	}

	if (state.color == GraphBuilder::ColorFamily::RGB) {
		if (state.subsample_w || state.subsample_h)
			error::throw_<error::UnsupportedSubsampling>("subsampled RGB image not supported");
		if (state.colorspace.matrix != colorspace::MatrixCoefficients::UNSPECIFIED && state.colorspace.matrix != colorspace::MatrixCoefficients::RGB)
			error::throw_<error::ColorFamilyMismatch>("RGB color family cannot have YUV matrix coefficients");
	}

	if (state.color == GraphBuilder::ColorFamily::YUV) {
		if (state.colorspace.matrix == colorspace::MatrixCoefficients::RGB)
			error::throw_<error::ColorFamilyMismatch>("YUV color family cannot have RGB matrix coefficients");
	}

	if (state.subsample_h > 1 && state.parity != GraphBuilder::FieldParity::PROGRESSIVE)
		error::throw_<error::UnsupportedSubsampling>("interlaced vertical subsampling greater than 2x is not supported");
	if (state.subsample_w > 2 || state.subsample_h > 2)
		error::throw_<error::UnsupportedSubsampling>("subsampling greater than 4x is not supported");

	if (state.width % (1 << state.subsample_w) || state.height % (1 << state.subsample_h))
		error::throw_<error::ImageNotDivisible>("image dimensions must be divisible by subsampling factor");

	if (state.depth > pixel_depth(state.type))
		error::throw_<error::BitDepthOverflow>("bit depth exceeds limits of type");
	if (!state.fullrange && state.depth < 8)
		error::throw_<error::BitDepthOverflow>("bit depth must be at least 8 for limited range");

	if (!std::isfinite(state.active_left) || !std::isfinite(state.active_top) || !std::isfinite(state.active_width) || !std::isfinite(state.active_height))
		error::throw_<error::InvalidImageSize>("active window must be finite");
	if (state.active_width <= 0 || state.active_height <= 0)
		error::throw_<error::InvalidImageSize>("active window must be positive");
}

} // namespace


/**
 * Holds a subgraph that can be inserted into an actual graph instance.
 */
class SubGraphBuilder {
private:
	std::vector<std::unique_ptr<graphengine::Filter>> m_filters;
	std::unique_ptr<graphengine::SubGraph> m_subgraph;
	graphengine::node_id m_source_ids[4];
	graphengine::node_id m_sink_ids[4];
public:
	SubGraphBuilder() :
		m_subgraph(std::make_unique<graphengine::SubGraphImpl>()),
		m_source_ids{},
		m_sink_ids{}
	{
		m_source_ids[0] = m_subgraph->add_source();
		m_source_ids[1] = m_subgraph->add_source();
		m_source_ids[2] = m_subgraph->add_source();
		m_source_ids[3] = m_subgraph->add_source();

		std::fill_n(m_sink_ids, 4, graphengine::null_node);
	}

	graphengine::node_id source_id(unsigned p) const { return m_source_ids[p]; }
	graphengine::node_id sink_id(unsigned p) const { return m_sink_ids[p]; }

	const graphengine::Filter *save_filter(std::unique_ptr<graphengine::Filter> filter)
	{
		m_filters.push_back(std::move(filter));
		return m_filters.back().get();
	}

	graphengine::node_id add_transform(const graphengine::Filter *filter, const graphengine::node_dep_desc deps[])
	{
		zassert_d(!!m_subgraph, "");
		return m_subgraph->add_transform(filter, deps);
	}

	void set_sink(unsigned num_planes, const graphengine::node_dep_desc deps[])
	{
		zassert_d(!!m_subgraph, "");

		for (unsigned p = 0; p < num_planes; ++p) {
			m_sink_ids[p] = m_subgraph->add_sink(deps[p]);
		}
	}

	std::pair<std::unique_ptr<graphengine::SubGraph>, std::shared_ptr<void>> release()
	{
		std::shared_ptr<void> opaque = std::make_shared<decltype(m_filters)>(std::move(m_filters));
		auto ret = std::make_pair(std::move(m_subgraph), std::move(opaque));
		*this = SubGraphBuilder{};
		return ret;
	}
};


struct GraphBuilder::internal_state {
	struct plane {
		unsigned width;
		unsigned height;
		PixelFormat format;
		double active_left;
		double active_top;
		double active_width;
		double active_height;

		plane() : width{}, height{}, format{}, active_left{}, active_top{}, active_width{}, active_height{} {}

		friend bool operator==(const plane &lhs, const plane &rhs)
		{
			return lhs.width == rhs.width &&
				lhs.height == rhs.height &&
				lhs.format == rhs.format &&
				lhs.active_left == rhs.active_left &&
				lhs.active_top == rhs.active_top &&
				lhs.active_width == rhs.active_width &&
				lhs.active_height == rhs.active_height;
		}

		friend bool operator!=(const plane &lhs, const plane &rhs)
		{
			return !(lhs == rhs);
		}
	};

	plane planes[4];
	ColorFamily color;
	colorspace::ColorspaceDefinition colorspace;
	AlphaType alpha;
private:
	void chroma_from_luma(unsigned subsample_w, unsigned subsample_h)
	{
		double subscale_w = 1.0 / (1UL << subsample_w);
		double subscale_h = 1.0 / (1UL << subsample_h);

		planes[PLANE_U] = planes[PLANE_Y];
		planes[PLANE_U].width >>= subsample_w;
		planes[PLANE_U].height >>= subsample_h;
		planes[PLANE_U].format.chroma = color == ColorFamily::YUV;
		planes[PLANE_U].active_left *= subscale_w;
		planes[PLANE_U].active_top *= subscale_h;
		planes[PLANE_U].active_width *= subscale_w;
		planes[PLANE_U].active_height *= subscale_h;

		planes[PLANE_V] = planes[PLANE_U];
	}

	void apply_pixel_siting(FieldParity parity, ChromaLocationW chroma_location_w, ChromaLocationH chroma_location_h)
	{
		planes[PLANE_Y].active_top -= luma_parity_offset(parity);

		if (color != ColorFamily::GREY) {
			double subscale_w = static_cast<double>(planes[PLANE_U].width) / planes[PLANE_Y].width;
			double subscale_h = static_cast<double>(planes[PLANE_U].height) / planes[PLANE_Y].height;
			double offset_w = chroma_offset_w(chroma_location_w, subscale_w);
			double offset_h = chroma_offset_parity_adjustment(parity, chroma_offset_h(chroma_location_h, subscale_h));

			planes[PLANE_U].active_left -= offset_w;
			planes[PLANE_U].active_top -= offset_h;
			planes[PLANE_V].active_left -= offset_w;
			planes[PLANE_V].active_top -= offset_h;
		}

		if (alpha != AlphaType::NONE)
			planes[PLANE_A].active_top -= luma_parity_offset(parity);
	}
public:
	internal_state() : planes{}, color{}, colorspace{}, alpha{} {}

	explicit internal_state(const state &state) :
		planes{},
		color{ state.color },
		colorspace(state.colorspace),
		alpha{ state.alpha }
	{
		planes[PLANE_Y].width = state.width;
		planes[PLANE_Y].height = state.height;
		planes[PLANE_Y].format.type = state.type;
		planes[PLANE_Y].format.depth = state.depth;
		planes[PLANE_Y].format.fullrange = pixel_is_integer(state.type) && state.fullrange;
		planes[PLANE_Y].format.chroma = false;
		planes[PLANE_Y].format.ycgco = colorspace.matrix == colorspace::MatrixCoefficients::YCGCO;
		planes[PLANE_Y].active_left = state.active_left;
		planes[PLANE_Y].active_top = state.active_top;
		planes[PLANE_Y].active_width = state.active_width;
		planes[PLANE_Y].active_height = state.active_height;

		if (color != ColorFamily::GREY)
			chroma_from_luma(state.subsample_w, state.subsample_h);

		if (alpha != AlphaType::NONE)
			alpha_from_luma();

		apply_pixel_siting(state.parity, state.chroma_location_w, state.chroma_location_h);
	}

	void chroma_from_luma_444()
	{
		planes[PLANE_U] = planes[PLANE_Y];
		planes[PLANE_U].format.chroma = color == ColorFamily::YUV;
		planes[PLANE_V] = planes[PLANE_U];
	}

	void alpha_from_luma()
	{
		planes[PLANE_A] = planes[PLANE_Y];
		planes[PLANE_A].format.fullrange = pixel_is_integer(planes[PLANE_A].format.type);
	}

	bool has_chroma() const { return color != ColorFamily::GREY; }

	bool has_alpha() const { return alpha != AlphaType::NONE; }

	friend bool operator==(const internal_state &lhs, const internal_state &rhs)
	{
		return lhs.color == rhs.color &&
			lhs.colorspace == rhs.colorspace &&
			lhs.alpha == rhs.alpha &&
			lhs.planes[PLANE_Y] == rhs.planes[PLANE_Y] &&
			(!lhs.has_chroma() || lhs.planes[PLANE_U] == rhs.planes[PLANE_U]) &&
			(!lhs.has_chroma() || lhs.planes[PLANE_V] == rhs.planes[PLANE_V]) &&
			(!lhs.has_alpha() || lhs.planes[PLANE_A] == rhs.planes[PLANE_A]);
	}

	friend bool operator!=(const internal_state &lhs, const internal_state &rhs)
	{
		return !(lhs == rhs);
	}
};


class GraphBuilder::impl {
private:
	enum class ConnectMode {
		LUMA,
		CHROMA,
		ALPHA,
	};

	SubGraphBuilder m_graph;
	std::array<graphengine::node_dep_desc, PLANE_NUM> m_ids;
	state m_source_state;
	internal_state m_state;
	bool m_requires_64b;

	internal_state make_float_444_state(const internal_state &state, bool include_alpha)
	{
		internal_state result = state;
		result.planes[PLANE_Y].format = PixelType::FLOAT;

		if (result.has_chroma())
			result.chroma_from_luma_444();
		if (result.has_alpha() && include_alpha)
			result.alpha_from_luma();

		return result;
	}

	template <class Func>
	void apply_mask(plane_mask mask, Func func)
	{
		for (int p = 0; p < PLANE_NUM; ++p) {
			if (mask[p])
				func(p);
		}
	}

	void attach_greyscale_filter(const graphengine::Filter *filter, plane_mask mask)
	{
		apply_mask(mask, [&](int p) { m_ids[p] = { m_graph.add_transform(filter, &m_ids[p]), 0 }; });
	}

	void check_is_444_float(bool check_alpha)
	{
		iassert(m_state.planes[PLANE_Y].format.type == PixelType::FLOAT);
		if (m_state.has_chroma()) {
			iassert(m_state.planes[PLANE_U].format.type == PixelType::FLOAT);
			iassert(m_state.planes[PLANE_V].format.type == PixelType::FLOAT);
		}
		if (check_alpha && m_state.has_alpha())
			iassert(m_state.planes[PLANE_A].format.type == PixelType::FLOAT);

		if (m_state.has_chroma()) {
			iassert(m_state.planes[0].width == m_state.planes[1].width && m_state.planes[0].height == m_state.planes[1].height);
			iassert(m_state.planes[0].width == m_state.planes[2].width && m_state.planes[0].height == m_state.planes[2].height);
		}
	}

	bool needs_colorspace(const internal_state &target)
	{
		colorspace::ColorspaceDefinition csp_in = m_state.colorspace;
		colorspace::ColorspaceDefinition csp_out = target.colorspace;

		if (csp_in == csp_out)
			return false;

		if (csp_in.transfer == csp_out.transfer && csp_in.primaries == csp_out.primaries) {
			// Can convert GREY-->YUV/RGB using value-init or dup filter.
			if (m_state.color == ColorFamily::GREY)
				return false;
			// Can convert YUV/GREY-->GREY by dropping chroma.
			if (m_state.color != ColorFamily::RGB && target.color == ColorFamily::GREY)
				return false;
		}

		return true;
	}

	bool needs_interpolation_plane(const internal_state &target, int p)
	{
		const internal_state::plane &source_plane = m_state.planes[p];
		const internal_state::plane &target_plane = target.planes[p];

		const auto phase = [](double x) { return std::modf(x, &x); };

		// 1. Is the resolution different?
		if (source_plane.active_width != target_plane.active_width || source_plane.active_height != target_plane.active_height)
			return true;

		// 2. Is the phase different?
		if (phase(source_plane.active_left) != phase(target_plane.active_left) || phase(source_plane.active_top) != phase(target_plane.active_top))
			return true;

		return false;
	}

	bool needs_interpolation(const internal_state &target)
	{
		if (needs_interpolation_plane(target, PLANE_Y))
			return true;
		if (m_state.has_chroma() && target.has_chroma() && needs_interpolation_plane(target, PLANE_U))
			return true;
		if (m_state.has_chroma() && target.has_chroma() && needs_interpolation_plane(target, PLANE_V))
			return true;
		if (m_state.has_alpha() && target.has_alpha() && needs_interpolation_plane(target, PLANE_A))
			return true;
		return false;
	}

	bool can_copy_tile(const internal_state &target, int p)
	{
		auto is_integer = [](double x) { return std::trunc(x) == x; };

		const internal_state::plane &src_plane = m_state.planes[p];
		const internal_state::plane &dst_plane = target.planes[p];

		// Input must be a subrectangle.
		if (src_plane.active_width == src_plane.width || src_plane.active_height == src_plane.height)
			return false;

		// Output must not be a subrectangle.
		if (dst_plane.active_width != dst_plane.width || dst_plane.active_height != dst_plane.height)
			return false;

		// Input subrectangle must be equal to output size.
		if (src_plane.active_width != dst_plane.width || src_plane.active_height != dst_plane.height)
			return false;

		double src_left = src_plane.active_left - dst_plane.active_left;
		double src_top = src_plane.active_top - dst_plane.active_top;

		// Input must be pixel-aligned.
		if (!is_integer(src_left) || !is_integer(src_top))
			return false;

		// Input subrectangle is not out of bounds.
		if (src_left < 0 || src_top < 0)
			return false;
		if (src_left + src_plane.active_width > src_plane.width || src_top + src_plane.active_height > src_plane.height)
			return false;

		return true;
	}

	bool needs_resize_plane(const internal_state &target, int p)
	{
		internal_state::plane plane = target.planes[p];
		plane.format = m_state.planes[p].format;
		return m_state.planes[p] != plane;
	}

	bool needs_resize(const internal_state &target)
	{
		if (needs_resize_plane(target, PLANE_Y))
			return true;
		if (m_state.has_chroma() && target.has_chroma() && needs_resize_plane(target, PLANE_U))
			return true;
		if (m_state.has_chroma() && target.has_chroma() && needs_resize_plane(target, PLANE_V))
			return true;
		if (m_state.has_alpha() && target.has_alpha() && needs_resize_plane(target, PLANE_A))
			return true;
		return false;
	}

	bool needs_premul(const internal_state &target)
	{
		if (m_state.alpha != AlphaType::STRAIGHT)
			return false;
		return target.alpha != AlphaType::STRAIGHT || needs_colorspace(target) || needs_interpolation(target);
	}

	void drop_plane(int p) { m_ids[p] = graphengine::null_dep; }

	void yuv_to_grey(FilterObserver &observer)
	{
		iassert(m_state.color == ColorFamily::YUV);

		observer.yuv_to_grey();

		drop_plane(PLANE_U);
		drop_plane(PLANE_V);
		m_state.color = ColorFamily::GREY;
	}

	void grey_to_rgb(colorspace::MatrixCoefficients matrix, FilterObserver &observer)
	{
		iassert(m_state.color == ColorFamily::GREY);

		observer.grey_to_rgb();

		m_ids[PLANE_U] = m_ids[PLANE_Y];
		m_ids[PLANE_V] = m_ids[PLANE_Y];

		m_state.color = ColorFamily::RGB;
		m_state.colorspace.matrix = matrix;
		m_state.chroma_from_luma_444();
	}

	void grey_to_yuv(const internal_state &target, FilterObserver &observer)
	{
		iassert(m_state.color == ColorFamily::GREY);

		observer.grey_to_yuv();

		PixelFormat format = m_state.planes[PLANE_Y].format;
		format.chroma = true;

		ValueInitializeFilter::value_type val;

		switch (format.type) {
		case PixelType::BYTE: val.b = 1U << (format.depth - 1); break;
		case PixelType::WORD: val.w = 1U << (format.depth - 1); break;
		case PixelType::HALF: val.w = 0; break;
		case PixelType::FLOAT: val.f = 0.0f; break;
		}

		auto filter = std::make_unique<ValueInitializeFilter>(
			target.planes[PLANE_U].width, target.planes[PLANE_U].height, format.type, val);
		graphengine::node_id id = m_graph.add_transform(m_graph.save_filter(std::move(filter)), nullptr);
		m_ids[PLANE_U] = { id, 0 };
		m_ids[PLANE_V] = { id, 0 };

		m_state.color = ColorFamily::YUV;
		m_state.colorspace.matrix = target.colorspace.matrix;
		m_state.planes[PLANE_U] = target.planes[PLANE_U];
		m_state.planes[PLANE_U].format = format;
		m_state.planes[PLANE_V] = target.planes[PLANE_V];
		m_state.planes[PLANE_V].format = format;
	}

	void premultiply(FilterObserver &observer)
	{
		iassert(m_state.alpha == AlphaType::STRAIGHT);
		check_is_444_float(true);

		observer.premultiply();

		auto filter = std::make_unique<PremultiplyFilter>(
			m_state.planes[PLANE_Y].width, m_state.planes[PLANE_Y].height);
		for (unsigned p = 0; p < (m_state.has_chroma() ? 3U : 1U); ++p) {
			graphengine::node_dep_desc deps[2] = { m_ids[p], m_ids[PLANE_A] };
			m_ids[p] = { m_graph.add_transform(filter.get(), deps), 0 };
		}
		m_graph.save_filter(std::move(filter));

		m_state.alpha = AlphaType::PREMULTIPLIED;
	}

	void unpremultiply(FilterObserver &observer)
	{
		iassert(m_state.alpha == AlphaType::PREMULTIPLIED);
		check_is_444_float(true);

		observer.unpremultiply();

		auto filter = std::make_unique<UnpremultiplyFilter>(
			m_state.planes[PLANE_Y].width, m_state.planes[PLANE_Y].height);
		for (unsigned p = 0; p < (m_state.has_chroma() ? 3U : 1U); ++p) {
			graphengine::node_dep_desc deps[2] = {m_ids[p], m_ids[PLANE_A]};
			m_ids[p] = { m_graph.add_transform(filter.get(), deps), 0 };
		}
		m_graph.save_filter(std::move(filter));

		m_state.alpha = AlphaType::STRAIGHT;
	}

	void add_opaque_alpha(AlphaType type, FilterObserver &observer)
	{
		iassert(m_state.alpha == AlphaType::NONE);

		observer.add_opaque();

		PixelFormat format = m_state.planes[PLANE_Y].format;
		ValueInitializeFilter::value_type val;

		switch (format.type) {
		case PixelType::BYTE: val.b = UINT8_MAX >> (8 - format.depth); break;
		case PixelType::WORD: val.w = UINT16_MAX >> (16 - format.depth); break;
		case PixelType::HALF: val.w = 0x3C00; break; // 1.0
		case PixelType::FLOAT: val.f = 1.0f; break;
		}

		auto filter = std::make_unique<ValueInitializeFilter>(
			m_state.planes[PLANE_Y].width, m_state.planes[PLANE_Y].height, format.type, val);
		m_ids[PLANE_A] = { m_graph.add_transform(m_graph.save_filter(std::move(filter)), nullptr), 0 };

		m_state.alpha = type;
		m_state.alpha_from_luma();
	}

	PixelFormat choose_resize_format(const internal_state &target, const params &params, int p)
	{
		if (params.unresize)
			return PixelType::FLOAT;

		bool supported[4] = { false, true, cpu_has_fast_f16(params.cpu), true };
		auto is_supported_type = [=](PixelType type) { return supported[static_cast<int>(type)]; };

		double src_pels = static_cast<double>(m_state.planes[p].width) * m_state.planes[p].height;
		double dst_pels = static_cast<double>(target.planes[p].width) * target.planes[p].height;

		PixelFormat src_format = m_state.planes[p].format;
		PixelFormat dst_format = target.planes[p].format;

		// If both formats are supported, pick the one that ends up converting the fewest pixels.
		if (is_supported_type(src_format.type) && is_supported_type(dst_format.type))
			return src_pels < dst_pels ? dst_format : src_format;

		// Pick the supported format.
		if (is_supported_type(src_format.type))
			return src_format;
		if (is_supported_type(dst_format.type))
			return dst_format;

		// Promote BYTE to WORD if dithering would not be required.
		if (is_supported_type(PixelType::WORD) && src_format.type == PixelType::BYTE && !src_format.fullrange)
			return { PixelType::WORD, 16, false, src_format.chroma, src_format.ycgco };

		// FLOAT is always supported.
		return PixelType::FLOAT;
	}

	void resize_plane(const internal_state &target, const params &params, FilterObserver &observer, plane_mask mask, int p)
	{
		if (!needs_resize_plane(target, p))
			return;

		const internal_state::plane &src_plane = m_state.planes[p];
		const internal_state::plane &dst_plane = target.planes[p];

		if (params.unresize) {
			if (src_plane.width != src_plane.active_width || src_plane.height != src_plane.active_height ||
			    dst_plane.width != dst_plane.active_width || dst_plane.height != dst_plane.active_height)
			{
				error::throw_<error::ResamplingNotAvailable>("unresize not supported for for given subregion");
			}
		}

		double scale_w = static_cast<double>(dst_plane.active_width) / src_plane.active_width;
		double scale_h = static_cast<double>(dst_plane.active_height) / src_plane.active_height;

		// Map active region in output image to the corresponding rectangle in input image.
		double shift_w = src_plane.active_left - dst_plane.active_left / scale_w;
		double shift_h = src_plane.active_top - dst_plane.active_top / scale_h;
		double subwidth = src_plane.active_width * (dst_plane.width / dst_plane.active_width);
		double subheight = src_plane.active_height * (dst_plane.height / dst_plane.active_height);

		std::unique_ptr<graphengine::Filter> first;
		std::unique_ptr<graphengine::Filter> second;

		if (params.unresize) {
			unresize::UnresizeConversion conv{ src_plane.width, src_plane.height, src_plane.format.type };
			conv.set_orig_width(dst_plane.width)
				.set_orig_height(dst_plane.height)
				.set_shift_w(shift_w)
				.set_shift_h(shift_h)
				.set_cpu(params.cpu);

			observer.unresize(conv, p);

			auto filter_list = conv.create();
			first = std::move(filter_list.first);
			second = std::move(filter_list.second);
		} else{
			resize::ResizeConversion conv{ src_plane.width, src_plane.height, src_plane.format.type };
			conv.set_depth(src_plane.format.depth)
				.set_filter(p == PLANE_U || p == PLANE_V ? params.filter_uv : params.filter)
				.set_dst_width(dst_plane.width)
				.set_dst_height(dst_plane.height)
				.set_shift_w(shift_w)
				.set_shift_h(shift_h)
				.set_subwidth(subwidth)
				.set_subheight(subheight)
				.set_cpu(params.cpu);

			observer.resize(conv, p);

			auto filter_list = conv.create();
			first = std::move(filter_list.first);
			second = std::move(filter_list.second);
		}

		if (first)
			attach_greyscale_filter(m_graph.save_filter(std::move(first)), mask);
		if (second)
			attach_greyscale_filter(m_graph.save_filter(std::move(second)), mask);

		apply_mask(mask, [&](int q)
		{
			PixelFormat format = m_state.planes[q].format;
			m_state.planes[q] = target.planes[q];
			m_state.planes[q].format = format;
		});
	}

	void convert_colorspace(const colorspace::ColorspaceDefinition &csp, const params &params, FilterObserver &observer)
	{
		iassert(m_state.color != ColorFamily::GREY);
		check_is_444_float(false);

		if (m_state.colorspace == csp)
			return;

		colorspace::ColorspaceConversion conv{ m_state.planes[0].width, m_state.planes[0].height };
		conv.set_csp_in(m_state.colorspace)
			.set_csp_out(csp)
			.set_approximate_gamma(params.approximate_gamma)
			.set_scene_referred(params.scene_referred)
			.set_cpu(params.cpu);
		if (!std::isnan(params.peak_luminance))
			conv.set_peak_luminance(params.peak_luminance);

		observer.colorspace(conv);

		auto filter = conv.create();
		if (filter) {
			graphengine::node_id id = m_graph.add_transform(m_graph.save_filter(std::move(filter)), m_ids.data());
			m_ids[PLANE_Y] = { id, 0 };
			m_ids[PLANE_U] = { id, 1 };
			m_ids[PLANE_V] = { id, 2 };
		}

		if (csp.matrix == colorspace::MatrixCoefficients::RGB) {
			m_state.color = ColorFamily::RGB;
			m_state.planes[PLANE_U].format.chroma = false;
			m_state.planes[PLANE_V].format.chroma = false;
		} else {
			m_state.color = ColorFamily::YUV;
			m_state.planes[PLANE_U].format.chroma = true;
			m_state.planes[PLANE_V].format.chroma = true;
		}
		m_state.colorspace = csp;
	}

	void convert_pixel_format(const PixelFormat &format, const params &params, FilterObserver &observer, plane_mask mask, int p)
	{
		if (m_state.planes[p].format == format)
			return;

		depth::DepthConversion conv{ m_state.planes[p].width, m_state.planes[p].height };
		conv.set_pixel_in(m_state.planes[p].format)
			.set_pixel_out(format)
			.set_dither_type(params.dither_type)
			.set_planes(mask)
			.set_cpu(params.cpu);

		observer.depth(conv, p);

		auto result = conv.create();
		apply_mask(mask, [&](int q)
		{
			if (result.filter_refs[q])
				m_ids[q] = { m_graph.add_transform(result.filter_refs[q], &m_ids[q]), 0 };
		});
		for (auto &&filter : result.filters) {
			m_graph.save_filter(std::move(filter));
		}

		apply_mask(mask, [&](int q) { m_state.planes[q].format = format; });
	}

	void connect_plane(const internal_state &target, const params &params, FilterObserver &observer, ConnectMode mode, bool reinterpret_range)
	{
		plane_mask mask{};
		internal_state tmp = target;
		bool reinterpreted = false;
		int p;

		if (mode == ConnectMode::LUMA) {
			mask = m_state.color == ColorFamily::RGB ? luma_planes | chroma_planes : luma_planes;
			p = PLANE_Y;
		} else if (mode == ConnectMode::CHROMA) {
			mask = chroma_planes;
			p = PLANE_U;
		} else {
			mask = alpha_planes;
			p = PLANE_A;
		}

		if (reinterpret_range) {
			PixelFormat src_format = m_state.planes[p].format;
			PixelFormat dst_format = tmp.planes[p].format;

			// Convert full-range BYTE to WORD by left-shifting if it would not affect the output.
			if (src_format.type == PixelType::BYTE && src_format.fullrange && dst_format.fullrange &&
			    src_format.depth == dst_format.depth)
			{
				apply_mask(mask, [&](int p) { m_state.planes[p].format.fullrange = false; });
				apply_mask(mask, [&](int p) { tmp.planes[p].format.fullrange = false; });
				reinterpreted = true;
			}
		}

		if (can_copy_tile(tmp, p)) {
			PixelFormat format = m_state.planes[p].format;
			unsigned left = static_cast<unsigned>(m_state.planes[p].active_left - tmp.planes[p].active_left);
			unsigned top = static_cast<unsigned>(m_state.planes[p].active_top - tmp.planes[p].active_top);

			observer.subrectangle(left, top, tmp.planes[p].width, tmp.planes[p].height, p);

			auto filter = std::make_unique<CopyRectFilter>(left, top, tmp.planes[p].width, tmp.planes[p].height, format.type);
			attach_greyscale_filter(m_graph.save_filter(std::move(filter)), mask);

			apply_mask(mask, [&](int q)
			{
				m_state.planes[q] = tmp.planes[q];
				m_state.planes[q].format = format;
			});

			iassert(!needs_resize_plane(tmp, p));
		}

		if (needs_resize_plane(tmp, p)) {
			PixelFormat format = choose_resize_format(tmp, params, p);
			convert_pixel_format(format, params, observer, mask, p);
			resize_plane(tmp, params, observer, mask, p);
		}

		if (m_state.planes[p].format != tmp.planes[p].format)
			convert_pixel_format(tmp.planes[p].format, params, observer, mask, p);

		// Undo temporary changes.
		if (reinterpreted)
			apply_mask(mask, [&](int p) { m_state.planes[p].format.fullrange = true; });

		iassert(m_state.planes[p] == target.planes[p]);
	}

	void connect_color_channels_planar(const internal_state &target, const params &params, FilterObserver &observer, bool reinterpret_range)
	{
		connect_plane(target, params, observer, ConnectMode::LUMA, reinterpret_range);

		if (m_state.color == ColorFamily::YUV)
			connect_plane(target, params, observer, ConnectMode::CHROMA, reinterpret_range);
	}

	void connect_color_channels(const internal_state &target, const params &params, FilterObserver &observer)
	{
		if (needs_colorspace(target)) {
			internal_state tmp = make_float_444_state(m_state, false);

			const internal_state &w = m_state.planes[PLANE_Y].width < target.planes[PLANE_Y].width ? m_state : target;
			const internal_state &h = m_state.planes[PLANE_Y].height < target.planes[PLANE_Y].height ? m_state : target;

			tmp.planes[PLANE_Y].width = w.planes[PLANE_Y].width;
			tmp.planes[PLANE_Y].height = h.planes[PLANE_Y].height;
			tmp.planes[PLANE_Y].active_left = w.planes[PLANE_Y].active_left;
			tmp.planes[PLANE_Y].active_top = h.planes[PLANE_Y].active_top;
			tmp.planes[PLANE_Y].active_width = w.planes[PLANE_Y].active_width;
			tmp.planes[PLANE_Y].active_height = h.planes[PLANE_Y].active_height;

			if (tmp.has_chroma())
				tmp.chroma_from_luma_444();

			connect_color_channels_planar(tmp, params, observer, false);

			if (!m_state.has_chroma()) {
				colorspace::MatrixCoefficients matrix =
					target.color == ColorFamily::RGB ? target.colorspace.matrix : colorspace::MatrixCoefficients::RGB;
				grey_to_rgb(matrix, observer);
			}

			convert_colorspace(target.colorspace, params, observer);
			iassert(m_state.colorspace == target.colorspace);
		}

		if (m_state.has_chroma() && !target.has_chroma())
			yuv_to_grey(observer);

		if (!m_state.has_chroma() && !target.has_chroma() &&
			m_state.colorspace.transfer == target.colorspace.transfer &&
			m_state.colorspace.primaries == target.colorspace.primaries)
		{
			m_state.colorspace.matrix = target.colorspace.matrix;
		}

		connect_color_channels_planar(target, params, observer, true);

		if (m_state.color == ColorFamily::GREY && target.color == ColorFamily::RGB)
			grey_to_rgb(target.colorspace.matrix, observer);
		if (m_state.color == ColorFamily::GREY && target.color == ColorFamily::YUV)
			grey_to_yuv(target, observer);

		iassert(m_state.color == target.color);
		iassert(m_state.colorspace == target.colorspace);
		iassert(m_state.planes[PLANE_Y] == target.planes[PLANE_Y]);
		iassert(!m_state.has_chroma() || m_state.planes[PLANE_U] == target.planes[PLANE_U]);
		iassert(!m_state.has_chroma() || m_state.planes[PLANE_V] == target.planes[PLANE_V]);
	}

	void connect_internal(const internal_state &target, const params &params, FilterObserver &observer)
	{
		if (needs_premul(target)) {
			internal_state::plane orig_alpha_plane = m_state.planes[PLANE_A];
			graphengine::node_dep_desc orig_alpha_node = m_ids[PLANE_A];

			internal_state tmp = make_float_444_state(m_state, true);
			connect_color_channels(tmp, params, observer);
			connect_plane(tmp, params, observer, ConnectMode::ALPHA, false);

			premultiply(observer);

			if (target.has_alpha() && target.planes[PLANE_A] == orig_alpha_plane) {
				m_ids[PLANE_A] = orig_alpha_node;
				m_state.planes[PLANE_A] = orig_alpha_plane;
			}
		}

		if (m_state.has_alpha() && !target.has_alpha()) {
			observer.discard_alpha();
			drop_plane(PLANE_A);
			m_state.alpha = AlphaType::NONE;
		}

		if (m_state.alpha == AlphaType::PREMULTIPLIED && target.alpha == AlphaType::STRAIGHT) {
			internal_state::plane orig_alpha_plane = m_state.planes[PLANE_A];
			graphengine::node_dep_desc orig_alpha_node = m_ids[PLANE_A];

			internal_state tmp = make_float_444_state(target, true);
			connect_color_channels(tmp, params, observer);
			connect_plane(tmp, params, observer, ConnectMode::ALPHA, false);

			unpremultiply(observer);

			if (target.has_alpha() && target.planes[PLANE_A] == orig_alpha_plane) {
				m_ids[PLANE_A] = orig_alpha_node;
				m_state.planes[PLANE_A] = orig_alpha_plane;
			}
		}

		connect_color_channels(target, params, observer);
		if (m_state.has_alpha()) {
			iassert(m_state.alpha == target.alpha);
			connect_plane(target, params, observer, ConnectMode::ALPHA, true);
		}

		if (!m_state.has_alpha() && target.has_alpha())
			add_opaque_alpha(target.alpha, observer);

		if (m_state != target)
			error::throw_<error::InternalError>("failed to connect graph");
	}
public:
	impl() :
		m_ids(),
		m_source_state{},
		m_state{},
		m_requires_64b{}
	{
		std::fill(m_ids.begin(), m_ids.end(), graphengine::null_dep);
	}

	void set_source(const state &source)
	{
		if (m_state.planes[0].width)
			error::throw_<error::InternalError>("graph already initialized");

		m_source_state = source;
		m_state = internal_state{ source };
		m_requires_64b = false;

		m_ids[PLANE_Y] = { m_graph.source_id(PLANE_Y), 0 };
		if (m_state.has_chroma()) {
			m_ids[PLANE_U] = { m_graph.source_id(PLANE_U), 0 };
			m_ids[PLANE_V] = { m_graph.source_id(PLANE_V), 0 };
		}
		if (m_state.has_alpha())
			m_ids[PLANE_A] = { m_graph.source_id(PLANE_A), 0 };
	}

	void connect(const state &target, const params &params, FilterObserver &observer)
	{
		if (!m_state.planes[0].width)
			error::throw_<error::InternalError>("graph not initialized");

		internal_state internal_target{ target };
		connect_internal(internal_target, params, observer);

#ifdef ZIMG_X86
		if (params.cpu == CPUClass::AUTO_64B || params.cpu >= CPUClass::X86_AVX512)
			m_requires_64b = true;
#endif
	}

	std::unique_ptr<SubGraph> build_subgraph()
	{
		if (!m_state.planes[0].width)
			error::throw_<error::InternalError>("graph not initialized");

		// Count input planes.
		std::array<graphengine::PlaneDescriptor, graphengine::NODE_MAX_PLANES> source_desc{};
		std::array<graphengine::node_id, graphengine::NODE_MAX_PLANES> source_ids;
		std::fill(source_ids.begin(), source_ids.end(), graphengine::null_node);

		{
			auto source_desc_it = source_desc.begin();
			auto source_ids_it = source_ids.begin();

			*source_desc_it++ = { m_source_state.width, m_source_state.height, zimg::pixel_size(m_source_state.type) };
			*source_ids_it++ = m_graph.source_id(PLANE_Y);
			if (m_source_state.color != ColorFamily::GREY) {
				*source_desc_it++ = { m_source_state.width >> m_source_state.subsample_w, m_source_state.height >> m_source_state.subsample_h, zimg::pixel_size(m_source_state.type) };
				*source_desc_it++ = { m_source_state.width >> m_source_state.subsample_w, m_source_state.height >> m_source_state.subsample_h, zimg::pixel_size(m_source_state.type) };
				*source_ids_it++ = m_graph.source_id(PLANE_U);
				*source_ids_it++ = m_graph.source_id(PLANE_V);
			}
			if (m_source_state.alpha != AlphaType::NONE) {
				*source_desc_it++ = { m_source_state.width, m_source_state.height, zimg::pixel_size(m_source_state.type) };
				*source_ids_it++ = m_graph.source_id(PLANE_A);
			}
		}

		// Count output planes.
		std::array<graphengine::node_dep_desc, graphengine::NODE_MAX_PLANES> sink_deps;
		std::fill(sink_deps.begin(), sink_deps.end(), graphengine::null_dep);

		unsigned num_sink_deps;
		{
			auto sink_deps_it = sink_deps.begin();

			*sink_deps_it++ = m_ids[PLANE_Y];
			if (m_state.has_chroma()) {
				*sink_deps_it++ = m_ids[PLANE_U];
				*sink_deps_it++ = m_ids[PLANE_V];
			}
			if (m_state.has_alpha())
				*sink_deps_it++ = m_ids[PLANE_A];

			num_sink_deps = static_cast<unsigned>(sink_deps_it - sink_deps.begin());
		}

		// Set sink planes.
		m_graph.set_sink(num_sink_deps, sink_deps.data());

		std::array<graphengine::node_id, graphengine::NODE_MAX_PLANES> sink_ids;
		std::fill(sink_ids.begin(), sink_ids.end(), graphengine::null_node);

		for (unsigned p = 0; p < num_sink_deps; ++p) {
			sink_ids[p] = m_graph.sink_id(p);
		}

		std::unique_ptr<graphengine::SubGraph> subgraph;
		std::shared_ptr<void> opaque;
		std::tie(subgraph, opaque) = m_graph.release();

		*this = impl();
		return std::make_unique<SubGraph>(std::move(subgraph), std::move(opaque), source_desc, source_ids, sink_ids);
	}
};


GraphBuilder::params::params() noexcept :
	filter{},
	filter_uv{},
	unresize{},
	dither_type{},
	peak_luminance{ NAN },
	approximate_gamma{},
	scene_referred{},
	cpu{ CPUClass::AUTO }
{
	static const resize::BicubicFilter bicubic;
	static const resize::BilinearFilter bilinear;

	filter = &bicubic;
	filter_uv = &bilinear;
}

GraphBuilder::GraphBuilder() try : m_impl(std::make_unique<impl>())
{
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
} catch (const std::bad_alloc &) {
	error::throw_<error::OutOfMemory>();
}

GraphBuilder::~GraphBuilder() = default;

GraphBuilder &GraphBuilder::set_source(const state &source) try
{
	validate_state(source);
	get_impl()->set_source(source);
	return *this;
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

GraphBuilder &GraphBuilder::connect(const state &target, const params *params, FilterObserver *observer) try
{
	static const GraphBuilder::params default_params;
	DefaultFilterObserver default_factory;

	validate_state(target);
	if (target.active_left != 0 || target.active_top != 0 || target.active_width != target.width || target.active_height != target.height)
		error::throw_<error::ResamplingNotAvailable>("active subregion not supported on target image");

	if (!params)
		params = &default_params;
	if (!observer)
		observer = &default_factory;

	get_impl()->connect(target, *params, *observer);
	return *this;
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

std::unique_ptr<SubGraph> GraphBuilder::build_subgraph() try
{
	return get_impl()->build_subgraph();
} catch (const graphengine::Exception &e) {
	rethrow_graphengine_exception(e);
} catch (const std::exception &e) {
	error::throw_<error::InternalError>(e.what());
}

std::unique_ptr<FilterGraph> GraphBuilder::build_graph()
{
	return build_subgraph()->build_full_graph();
}

} // namespace zimg::graph
