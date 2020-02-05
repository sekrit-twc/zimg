#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <new>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "resize/filter.h"
#include "basic_filter.h"
#include "filtergraph2.h"
#include "graphbuilder2.h"

namespace zimg {
namespace graph {

namespace {

#ifndef ZIMG_UNSAFE_IMAGE_SIZE
#include <limits>
constexpr size_t IMAGE_DIMENSION_MAX = static_cast<size_t>(1U) << (std::numeric_limits<size_t>::digits / 2 - 2);
#else
constexpr size_t IMAGE_DIMENSION_MAX = ~static_cast<size_t>(0);
#endif // ZIMG_UNSAFE_IMAGE_SIZE

double chroma_shift_raw(GraphBuilder2::ChromaLocationW loc, GraphBuilder2::FieldParity)
{
	if (loc == GraphBuilder2::ChromaLocationW::LEFT)
		return -0.5;
	else
		return 0.0;
}

double chroma_shift_raw(GraphBuilder2::ChromaLocationH loc, GraphBuilder2::FieldParity parity)
{
	double shift;

	if (loc == GraphBuilder2::ChromaLocationH::TOP)
		shift = -0.5;
	else if (loc == GraphBuilder2::ChromaLocationH::BOTTOM)
		shift = 0.5;
	else
		shift = 0;

	if (parity == GraphBuilder2::FieldParity::TOP)
		shift = (shift - 0.5) / 2.0;
	else if (parity == GraphBuilder2::FieldParity::BOTTOM)
		shift = (shift + 0.5) / 2.0;

	return shift;
}

template <class T>
double chroma_shift_factor(T loc_in, T loc_out, unsigned subsample_in, unsigned subsample_out, GraphBuilder2::FieldParity parity, unsigned src_dim, unsigned dst_dim)
{
	double shift = 0.0;
	double sub_scale = 1.0 / (1 << subsample_in);

	if (subsample_in)
		shift -= sub_scale * chroma_shift_raw(loc_in, parity);
	if (subsample_out)
		shift += sub_scale * chroma_shift_raw(loc_out, parity) * src_dim / dst_dim;

	return shift;
}

double luma_shift_factor(GraphBuilder2::FieldParity parity, unsigned src_height, unsigned dst_height)
{
	double shift = 0.0;

	if (parity == GraphBuilder2::FieldParity::TOP)
		shift = -0.25;
	else if (parity == GraphBuilder2::FieldParity::BOTTOM)
		shift = 0.25;

	return shift * src_height / dst_height - shift;
}


bool is_greyscale(const GraphBuilder2::state &state) { return state.color == GraphBuilder2::ColorFamily::GREY; }

bool is_rgb(const GraphBuilder2::state &state) { return state.color == GraphBuilder2::ColorFamily::RGB; }

bool is_yuv(const GraphBuilder2::state &state) { return state.color == GraphBuilder2::ColorFamily::YUV; }

bool is_color(const GraphBuilder2::state &state) { return !is_greyscale(state); }

bool is_ycgco(const GraphBuilder2::state &state) { return state.colorspace.matrix == colorspace::MatrixCoefficients::YCGCO; }

bool has_alpha(const GraphBuilder2::state &state) { return state.alpha != GraphBuilder2::AlphaType::NONE; }

void validate_state(const GraphBuilder2::state &state)
{
	if (!state.width || !state.height)
		error::throw_<error::InvalidImageSize>("image dimensions must be non-zero");
	if (state.width > IMAGE_DIMENSION_MAX || state.height > IMAGE_DIMENSION_MAX)
		error::throw_<error::InvalidImageSize>("image dimensions exceed implementation limit");
	if (state.width > pixel_max_width(state.type))
		error::throw_<error::InvalidImageSize>("image width exceeds memory addressing limit");

	if (is_greyscale(state)) {
		if (state.subsample_w || state.subsample_h)
			error::throw_<error::GreyscaleSubsampling>("cannot subsample greyscale image");
		if (state.colorspace.matrix == colorspace::MatrixCoefficients::RGB)
			error::throw_<error::ColorFamilyMismatch>("GREY color family cannot have RGB matrix coefficients");
	}

	if (is_rgb(state)) {
		if (state.subsample_w || state.subsample_h)
			error::throw_<error::UnsupportedSubsampling>("subsampled RGB image not supported");
		if (state.colorspace.matrix != colorspace::MatrixCoefficients::UNSPECIFIED && state.colorspace.matrix != colorspace::MatrixCoefficients::RGB)
			error::throw_<error::ColorFamilyMismatch>("RGB color family cannot have YUV matrix coefficients");
	}

	if (is_yuv(state)) {
		if (state.colorspace.matrix == colorspace::MatrixCoefficients::RGB)
			error::throw_<error::ColorFamilyMismatch>("YUV color family cannot have RGB matrix coefficients");
	}

	if (state.subsample_h > 1 && state.parity != GraphBuilder2::FieldParity::PROGRESSIVE)
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

bool needs_colorspace(const GraphBuilder2::state &source, const GraphBuilder2::state &target)
{
	auto csp_in = source.colorspace;
	auto csp_out = target.colorspace;

	if (is_greyscale(target) && !is_rgb(source))
		csp_in.matrix = csp_out.matrix;

	return csp_in != csp_out;
}

bool needs_depth(const GraphBuilder2::state &source, const GraphBuilder2::state &target)
{
	return PixelFormat{ source.type, source.depth, source.fullrange } != PixelFormat{ target.type, target.depth, target.fullrange };
}

bool needs_resize(const GraphBuilder2::state &source, const GraphBuilder2::state &target)
{
	bool result = source.width != target.width ||
		          source.height != target.height ||
		          source.active_left != target.active_left ||
		          source.active_top != target.active_top ||
		          source.active_width != target.active_width ||
		          source.active_height != target.active_height;

	if (is_color(source) || is_color(target)) {
		result = result ||
		         source.subsample_w != target.subsample_w ||
			     source.subsample_h != target.subsample_h ||
		         (source.subsample_w && source.chroma_location_w != target.chroma_location_w) ||
			     (source.subsample_h && source.chroma_location_h != target.chroma_location_h);
	};

	return result;
}

} // namespace


auto DefaultFilterFactory2::create_colorspace(const colorspace::ColorspaceConversion &conv) -> filter_list
{
	std::unique_ptr<ImageFilter> filters[1] = { conv.create() };
	return{ std::make_move_iterator(filters), std::make_move_iterator(filters + 1) };
}

auto DefaultFilterFactory2::create_depth(const depth::DepthConversion &conv) -> filter_list
{
	std::unique_ptr<ImageFilter> filters[1] = { conv.create() };
	return{ std::make_move_iterator(filters), std::make_move_iterator(filters + 1) };
}

auto DefaultFilterFactory2::create_resize(const resize::ResizeConversion &conv) -> filter_list
{
	auto filter_pair = conv.create();
	filter_list list;

	if (filter_pair.first)
		list.emplace_back(std::move(filter_pair.first));
	if (filter_pair.second)
		list.emplace_back(std::move(filter_pair.second));

	return list;
}

auto DefaultFilterFactory2::create_unresize(const unresize::UnresizeConversion &conv) -> filter_list
{
	auto filter_pair = conv.create();
	filter_list list;

	if (filter_pair.first)
		list.emplace_back(std::move(filter_pair.first));
	if (filter_pair.second)
		list.emplace_back(std::move(filter_pair.second));

	return list;
}


GraphBuilder2::params::params() noexcept :
	unresize{},
	dither_type{},
	peak_luminance{ NAN },
	approximate_gamma{},
	scene_referred{},
	cpu{}
{}

struct GraphBuilder2::resize_spec {
	unsigned width;
	unsigned height;
	unsigned subsample_w;
	unsigned subsample_h;
	double shift_w;
	double shift_h;
	double subwidth;
	double subheight;
	ChromaLocationW chroma_location_w;
	ChromaLocationH chroma_location_h;

	resize_spec() = default;

	explicit resize_spec(const state &state) :
		width{ state.width },
		height{ state.height },
		subsample_w{ state.subsample_w },
		subsample_h{ state.subsample_h },
		shift_w{ state.active_left },
		shift_h{ state.active_top },
		subwidth{ state.active_width },
		subheight{ state.active_height },
		chroma_location_w{ state.chroma_location_w },
		chroma_location_h{ state.chroma_location_h }
	{}
};

GraphBuilder2::GraphBuilder2() noexcept : m_state{}, m_plane_ids(null_ids) {}

GraphBuilder2::~GraphBuilder2() = default;

GraphBuilder2::state GraphBuilder2::make_alpha_state(const state &s)
{
	state result = s;
	result.color = ColorFamily::GREY;
	result.colorspace = { colorspace::MatrixCoefficients::UNSPECIFIED, colorspace::TransferCharacteristics::UNSPECIFIED, colorspace::ColorPrimaries::UNSPECIFIED };
	result.fullrange = true;
	result.alpha = AlphaType::NONE;

	return result;
}

void GraphBuilder2::attach_greyscale_filter(std::shared_ptr<ImageFilter> filter, int plane, bool dep)
{
	id_map deps = null_ids;
	plane_mask mask{};

	deps[plane] = dep ? m_plane_ids[plane] : -1;
	mask[plane] = true;

	m_plane_ids[plane] = m_graph->attach_filter(std::move(filter), deps, mask);
}

void GraphBuilder2::attach_color_filter(std::shared_ptr<ImageFilter> filter)
{
	id_map deps = null_ids;
	plane_mask mask{};

	deps[PLANE_Y] = m_plane_ids[PLANE_Y];
	deps[PLANE_U] = m_plane_ids[PLANE_U];
	deps[PLANE_V] = m_plane_ids[PLANE_V];

	mask[PLANE_Y] = true;
	mask[PLANE_U] = true;
	mask[PLANE_V] = true;

	node_id id = m_graph->attach_filter(std::move(filter), deps, mask);
	m_plane_ids[PLANE_Y] = id;
	m_plane_ids[PLANE_U] = id;
	m_plane_ids[PLANE_V] = id;
}

void GraphBuilder2::convert_colorspace(const colorspace::ColorspaceDefinition &colorspace, const params *params, FilterFactory2 *factory)
{
	zassert_d(!is_greyscale(m_state), "expected color image");

	if (m_state.colorspace == colorspace)
		return;

	CPUClass cpu = params ? params->cpu : CPUClass::AUTO;

	colorspace::ColorspaceConversion conv{ m_state.width, m_state.height };
	conv.set_csp_in(m_state.colorspace)
		.set_csp_out(colorspace)
		.set_cpu(cpu);

	if (params) {
		if (!std::isnan(params->peak_luminance))
			conv.set_peak_luminance(params->peak_luminance);
		conv.set_approximate_gamma(params->approximate_gamma);
		conv.set_scene_referred(params->scene_referred);
	}

	for (auto &&filter : factory->create_colorspace(conv)) {
		attach_color_filter(std::move(filter));
	}

	m_state.color = colorspace.matrix == colorspace::MatrixCoefficients::RGB ? ColorFamily::RGB : ColorFamily::YUV;
	m_state.colorspace = colorspace;
}

void GraphBuilder2::convert_depth(state *state, const PixelFormat &format, const params *params, FilterFactory2 *factory, bool alpha)
{
	PixelFormat basic_format{ state->type, state->depth, state->fullrange };

	if (basic_format == format)
		return;

	CPUClass cpu = params ? params->cpu : CPUClass::AUTO;
	depth::DitherType dither_type = params ? params->dither_type : depth::DitherType::NONE;

	// (1) Luma or alpha plane.
	{
		depth::DepthConversion conv{ state->width, state->height };
		conv.set_pixel_in(basic_format)
		    .set_pixel_out(format)
		    .set_dither_type(dither_type)
		    .set_cpu(cpu);

		for (auto &&filter : factory->create_depth(conv)) {
			if (is_rgb(*state)) {
				std::shared_ptr<ImageFilter> shared{ std::move(filter) };
				attach_greyscale_filter(shared, PLANE_Y);
				attach_greyscale_filter(shared, PLANE_U);
				attach_greyscale_filter(shared, PLANE_V);
			} else {
				attach_greyscale_filter(std::move(filter), alpha ? PLANE_A : PLANE_Y);
			}
		}
	}

	// (2) Chroma planes.
	if (is_yuv(*state)) {
		PixelFormat source_chroma_format = basic_format;
		source_chroma_format.chroma = true;
		source_chroma_format.ycgco = is_ycgco(*state);

		PixelFormat target_chroma_format = format;
		target_chroma_format.chroma = true;
		target_chroma_format.ycgco = is_ycgco(*state);

		depth::DepthConversion conv{ state->width >> state->subsample_w, state->height >> state->subsample_h };
		conv.set_pixel_in(source_chroma_format)
		    .set_pixel_out(target_chroma_format)
		    .set_dither_type(dither_type)
		    .set_cpu(cpu);

		for (auto &&filter : factory->create_depth(conv)) {
			std::shared_ptr<ImageFilter> shared{ std::move(filter) };
			attach_greyscale_filter(shared, PLANE_U);
			attach_greyscale_filter(shared, PLANE_V);
		}
	}

	state->type = format.type;
	state->depth = format.depth;
	state->fullrange = format.fullrange;
}

void GraphBuilder2::convert_resize(state *state, const resize_spec &spec, const params *params, FilterFactory2 *factory, bool alpha)
{
	unsigned subsample_w = is_color(*state) ? spec.subsample_w : 0;
	unsigned subsample_h = is_color(*state) ? spec.subsample_h : 0;
	ChromaLocationW chroma_loc_w = subsample_w ? spec.chroma_location_w : ChromaLocationW::CENTER;
	ChromaLocationH chroma_loc_h = subsample_h ? spec.chroma_location_h : ChromaLocationH::CENTER;

	bool image_shifted = spec.shift_w != 0.0 || spec.shift_h != 0.0 || state->width != spec.subwidth || state->height != spec.subheight;

	if (state->width == spec.width &&
		state->height == spec.height &&
		state->subsample_w == subsample_w &&
		state->subsample_h == subsample_h &&
		state->chroma_location_w == chroma_loc_w &&
		state->chroma_location_h == chroma_loc_h && !image_shifted)
	{
		return;
	}

	// Default filter instances.
	resize::BicubicFilter bicubic;
	resize::BilinearFilter bilinear;

	const resize::Filter *resample_filter = params ? params->filter.get() : &bicubic;
	const resize::Filter *resample_filter_uv = params ? params->filter_uv.get() : &bilinear;
	bool unresize = params && params->unresize;
	CPUClass cpu = params ? params->cpu : CPUClass::AUTO;

	if (unresize && (spec.subwidth != state->width || spec.subheight != state->height))
		error::throw_<error::ResamplingNotAvailable>("unresize not supported for given subregion");

	// (1) Luma or alpha plane.
	if (state->width != spec.width || state->height != spec.height || image_shifted) {
		double extra_shift_h = luma_shift_factor(state->parity, state->height, spec.height);

		FilterFactory2::filter_list filters;
		if (unresize) {
			unresize::UnresizeConversion conv{ state->width, state->height, state->type };
			conv.set_orig_width(spec.width)
			    .set_orig_height(spec.height)
			    .set_shift_w(spec.shift_w)
			    .set_shift_h(spec.shift_h + extra_shift_h)
			    .set_cpu(cpu);
			filters = factory->create_unresize(conv);
		} else {
			resize::ResizeConversion conv{ state->width, state->height, state->type };
			conv.set_depth(state->depth)
			    .set_filter(resample_filter)
			    .set_dst_width(spec.width)
			    .set_dst_height(spec.height)
			    .set_shift_w(spec.shift_w)
			    .set_shift_h(spec.shift_h + extra_shift_h)
			    .set_subwidth(spec.subwidth)
			    .set_subheight(spec.subheight)
			    .set_cpu(cpu);
			filters = factory->create_resize(conv);
		}

		for (auto &&filter : filters) {
			if (is_rgb(*state)) {
				std::shared_ptr<ImageFilter> shared{ std::move(filter) };
				attach_greyscale_filter(shared, PLANE_Y);
				attach_greyscale_filter(shared, PLANE_U);
				attach_greyscale_filter(shared, PLANE_V);
			} else {
				attach_greyscale_filter(std::move(filter), alpha ? PLANE_A : PLANE_Y);
			}
		}
	}

	// (2) Chroma planes.
	if (is_yuv(*state) && (
	    (state->width >> state->subsample_w != spec.width >> subsample_w) ||
	    (state->height >> state->subsample_h != spec.height >> subsample_h) ||
	    ((state->subsample_w || subsample_w) && state->chroma_location_w != chroma_loc_w) ||
	    ((state->subsample_h || subsample_h) && state->chroma_location_h != chroma_loc_h) ||
	    image_shifted))
	{
		unsigned width_in = state->width >> state->subsample_w;
		unsigned height_in = state->height >> state->subsample_h;
		unsigned width_out = spec.width >> subsample_w;
		unsigned height_out = spec.height >> subsample_h;

		double extra_shift_w = chroma_shift_factor(
			state->chroma_location_w, chroma_loc_w, state->subsample_w, subsample_w, state->parity, state->width, spec.width);
		double extra_shift_h = chroma_shift_factor(
			state->chroma_location_h, chroma_loc_h, state->subsample_h, subsample_h, state->parity, state->height, spec.height);

		FilterFactory2::filter_list filters;
		if (unresize) {
			unresize::UnresizeConversion conv{ width_in, height_in, state->type };
			conv.set_orig_width(width_out)
				.set_orig_height(height_out)
				.set_shift_w(spec.shift_w / (1 << state->subsample_w) + extra_shift_w)
				.set_shift_h(spec.shift_h / (1 << state->subsample_h) + extra_shift_h)
				.set_cpu(cpu);
			filters = factory->create_unresize(conv);
		} else {
			resize::ResizeConversion conv{ width_in, height_in, state->type };
			conv.set_depth(state->depth)
				.set_filter(resample_filter_uv)
				.set_dst_width(width_out)
				.set_dst_height(height_out)
				.set_shift_w(spec.shift_w / (1 << state->subsample_w) + extra_shift_w)
				.set_shift_h(spec.shift_h / (1 << state->subsample_h) + extra_shift_h)
				.set_subwidth(spec.subwidth / (1 << state->subsample_w))
				.set_subheight(spec.subheight / (1 << state->subsample_h))
				.set_cpu(cpu);
			filters = factory->create_resize(conv);
		}

		for (auto &&filter : filters) {
			std::shared_ptr<ImageFilter> shared{ std::move(filter) };
			attach_greyscale_filter(shared, PLANE_U);
			attach_greyscale_filter(shared, PLANE_V);
		}
	}

	state->width = spec.width;
	state->height = spec.height;
	state->subsample_w = subsample_w;
	state->subsample_h = subsample_h;
	state->chroma_location_w = chroma_loc_w;
	state->chroma_location_h = chroma_loc_h;
	state->active_left = 0.0;
	state->active_top = 0.0;
	state->active_width = spec.width;
	state->active_height = spec.height;
}

void GraphBuilder2::add_opaque_alpha()
{
	ValueInitializeFilter::value_type val;

	switch (m_state.type) {
	case PixelType::BYTE:
		val.b = UINT8_MAX >> (8 - m_state.depth);
		break;
	case PixelType::WORD:
		val.w = UINT16_MAX >> (16 - m_state.depth);
		break;
	case PixelType::HALF:
		val.w = 0x3C00;
		break;
	case PixelType::FLOAT:
		val.f = 1.0f;
		break;
	}

	auto filter = std::make_shared<ValueInitializeFilter>(m_state.width, m_state.height, m_state.type, val);
	attach_greyscale_filter(std::move(filter), PLANE_A, false);
}

void GraphBuilder2::discard_chroma()
{
	zassert_d(is_yuv(m_state), "can not drop chroma planes from RGB image");
	m_plane_ids[PLANE_U] = -1;
	m_plane_ids[PLANE_V] = -1;
	m_state.color = ColorFamily::GREY;
	m_state.subsample_w = 0;
	m_state.subsample_h = 0;
}

void GraphBuilder2::grey_to_color(ColorFamily color, unsigned subsample_w, unsigned subsample_h, ChromaLocationW chroma_loc_w, ChromaLocationH chroma_loc_h)
{
	if (color == ColorFamily::RGB) {
		zassert_d(!subsample_w && !subsample_h, "RGB can not be subsampled");

		id_map deps = null_ids;
		plane_mask mask{};

		deps[PLANE_Y] = m_plane_ids[PLANE_Y];
		mask[PLANE_U] = true;
		mask[PLANE_V] = true;

		auto filter = std::make_shared<RGBExtendFilter>(m_state.width, m_state.height, m_state.type);
		node_id id = m_graph->attach_filter(std::move(filter), deps, mask);
		m_plane_ids[PLANE_U] = id;
		m_plane_ids[PLANE_V] = id;
	} else {
		ValueInitializeFilter::value_type val;

		switch (m_state.type) {
		case PixelType::BYTE:
			val.b = 1U << (m_state.depth - 1);
			break;
		case PixelType::WORD:
			val.w = 1U << (m_state.depth - 1);
			break;
		case PixelType::HALF:
			val.w = 0;
			break;
		case PixelType::FLOAT:
			val.f = 0.0f;
			break;
		}

		auto filter = std::make_shared<ValueInitializeFilter>(m_state.width >> subsample_w, m_state.height >> subsample_h, m_state.type, val);
		attach_greyscale_filter(filter, PLANE_U, false);
		attach_greyscale_filter(filter, PLANE_V, false);
	}

	m_state.color = color;
	m_state.subsample_w = subsample_w;
	m_state.subsample_h = subsample_h;
	m_state.chroma_location_w = chroma_loc_w;
	m_state.chroma_location_h = chroma_loc_h;
}

void GraphBuilder2::premultiply(const params *params, FilterFactory2 *factory)
{
	zassert_d(m_state.alpha == AlphaType::STRAIGHT, "must be straight alpha");

	state alpha_state = make_alpha_state(m_state);
	convert_depth(&m_state, PixelType::FLOAT, params, factory, false);
	convert_depth(&alpha_state, PixelType::FLOAT, params, factory, true);

	resize_spec resize{ m_state };
	resize.subsample_w = 0;
	resize.subsample_h = 0;
	convert_resize(&m_state, resize, params, factory, false);

	auto filter = std::make_shared<PremultiplyFilter>(m_state.width, m_state.height, is_color(m_state));

	id_map deps = null_ids;
	deps[PLANE_Y] = m_plane_ids[PLANE_Y];
	deps[PLANE_U] = is_color(m_state) ? m_plane_ids[PLANE_U] : -1;
	deps[PLANE_V] = is_color(m_state) ? m_plane_ids[PLANE_V] : -1;
	deps[PLANE_A] = m_plane_ids[PLANE_A];

	plane_mask mask{};
	mask[PLANE_Y] = true;
	mask[PLANE_U] = is_color(m_state);
	mask[PLANE_V] = is_color(m_state);

	node_id id = m_graph->attach_filter(std::move(filter), deps, mask);
	m_plane_ids[PLANE_Y] = id;
	m_plane_ids[PLANE_U] = is_color(m_state) ? id : -1;
	m_plane_ids[PLANE_V] = is_color(m_state) ? id : -1;

	m_state.alpha = AlphaType::PREMULTIPLED;
}

void GraphBuilder2::unpremultiply(const params *params, FilterFactory2 *factory)
{
	zassert_d(m_state.alpha == AlphaType::PREMULTIPLED, "must be premultiplied");

	state alpha_state = make_alpha_state(m_state);
	convert_depth(&m_state, PixelType::FLOAT, params, factory, false);
	convert_depth(&alpha_state, PixelType::FLOAT, params, factory, true);

	resize_spec resize{ m_state };
	resize.subsample_w = 0;
	resize.subsample_h = 0;
	convert_resize(&m_state, resize, params, factory, false);

	auto filter = std::make_shared<UnpremultiplyFilter>(m_state.width, m_state.height, is_color(m_state));

	id_map deps = null_ids;
	deps[PLANE_Y] = m_plane_ids[PLANE_Y];
	deps[PLANE_U] = is_color(m_state) ? m_plane_ids[PLANE_U] : -1;
	deps[PLANE_V] = is_color(m_state) ? m_plane_ids[PLANE_V] : -1;
	deps[PLANE_A] = m_plane_ids[PLANE_A];

	plane_mask mask{};
	mask[PLANE_Y] = true;
	mask[PLANE_U] = is_color(m_state);
	mask[PLANE_V] = is_color(m_state);

	node_id id = m_graph->attach_filter(std::move(filter), deps, mask);
	m_plane_ids[PLANE_Y] = id;
	m_plane_ids[PLANE_U] = is_color(m_state) ? id : -1;
	m_plane_ids[PLANE_V] = is_color(m_state) ? id : -1;

	m_state.alpha = AlphaType::STRAIGHT;
}

void GraphBuilder2::connect_color_channels(const state &target, const params *params, FilterFactory2 *factory)
{
	bool fast_f16 = cpu_has_fast_f16(params ? params->cpu : CPUClass::NONE);

	// (1) Convert to target colorspace.
	if (needs_colorspace(m_state, target)) {
		// Determine whether resampling is needed.
		resize_spec spec{ m_state };
		spec.subsample_w = 0;
		spec.subsample_h = 0;

		if ((m_state.subsample_w || m_state.subsample_h) &&
			(!target.subsample_w && !target.subsample_h)) {
			spec.width = target.width;
			spec.height = target.height;
		} else {
			spec.width = std::min(m_state.width, target.width);
			spec.height = std::min(m_state.height, target.height);
		}

		// (1.1) Convert to float.
		if (m_state.type != PixelType::FLOAT)
			convert_depth(&m_state, PixelType::FLOAT, params, factory, false);

		// (1.2) Resize to optimal dimensions. Chroma subsampling is also handled here.
		convert_resize(&m_state, spec, params, factory, false);

		// (1.3) Add fake chroma planes if needed.
		if (is_greyscale(m_state)) {
			grey_to_color(ColorFamily::RGB, 0, 0, ChromaLocationW::CENTER, ChromaLocationH::CENTER);
			m_state.colorspace.matrix = colorspace::MatrixCoefficients::RGB;
		}

		// (1.4) Convert colorspace.
		convert_colorspace(target.colorspace, params, factory);
		zassert_d(!needs_colorspace(m_state, target), "conversion did not apply");
	}

	// (2) Drop chroma planes (including fake planes from (1)).
	if (!is_greyscale(m_state) && is_greyscale(target))
		discard_chroma();

	// (3) Resize to target dimensions.
	if (needs_resize(m_state, target)) {
		// (3.1) Convert to compatible pixel type, attempting to minimize total conversions.
		// If neither the source nor target pixel format is directly supported, select a different format.
		// Direct operation on half-precision is slightly slower, so avoid it if the target is not also half.
		if (params && params->unresize)
			convert_depth(&m_state, PixelType::FLOAT, params, factory, false);
		else if (target.type == PixelType::WORD)
			convert_depth(&m_state, PixelFormat{ target.type, target.depth, target.fullrange, false, is_ycgco(target) }, params, factory, false);
		else if (target.type == PixelType::HALF && fast_f16)
			convert_depth(&m_state, PixelType::HALF, params, factory, false);
		else if (target.type == PixelType::FLOAT)
			convert_depth(&m_state, PixelType::FLOAT, params, factory, false);
		else if (m_state.type == PixelType::BYTE)
			convert_depth(&m_state, PixelFormat{ PixelType::WORD, 16, false, false, is_ycgco(target) }, params, factory, false);
		else if (m_state.type == PixelType::HALF && (target.type != PixelType::HALF || !fast_f16))
			convert_depth(&m_state, PixelType::FLOAT, params, factory, false);

		// (3.2) Resize image.
		resize_spec spec{ target };
		spec.shift_w = m_state.active_left;
		spec.shift_h = m_state.active_top;
		spec.subwidth = m_state.active_width;
		spec.subheight = m_state.active_height;
		spec.chroma_location_w = target.chroma_location_w;
		spec.chroma_location_h = target.chroma_location_h;

		convert_resize(&m_state, spec, params, factory, false);
		zassert_d(!needs_resize(m_state, target), "conversion did not apply");
	}

	// (4) Convert to target pixel format.
	if (needs_depth(m_state, target)) {
		convert_depth(&m_state, PixelFormat{ target.type, target.depth, target.fullrange }, params, factory, false);
		zassert_d(!needs_depth(m_state, target), "conversion did not apply");
	}

	// (5) Add fake chroma planes.
	if (is_greyscale(m_state) && !is_greyscale(target))
		grey_to_color(target.color, target.subsample_w, target.subsample_h, target.chroma_location_w, target.chroma_location_h);
}

void GraphBuilder2::connect_alpha_channel(const state &orig, const state &target, const params *params, FilterFactory2 *factory)
{
	zassert_d(has_alpha(m_state) && has_alpha(target), "alpha channel missing");

	state source_alpha = make_alpha_state(orig);
	state target_alpha = make_alpha_state(target);

	bool fast_f16 = cpu_has_fast_f16(params ? params->cpu : CPUClass::NONE);

	// (1) Resize plane to target dimensions.
	if (needs_resize(source_alpha, target_alpha)) {
		// (1.1) Convert to compatible pixel type.
		if (source_alpha.type != PixelType::FLOAT)
			convert_depth(&source_alpha, PixelType::FLOAT, params, factory, true);

		// (1.1) Convert to compatible pixel type, attempting to minimize total conversions.
		// If neither the source nor target pixel format is directly supported, select a different format.
		// Direct operation on half-precision is slightly slower, so avoid it if the target is not also half.
		if (params && params->unresize)
			convert_depth(&source_alpha, PixelType::FLOAT, params, factory, true);
		else if (target_alpha.type == PixelType::WORD)
			convert_depth(&source_alpha, PixelFormat{ target_alpha.type, target_alpha.depth, target_alpha.fullrange }, params, factory, true);
		else if (target_alpha.type == PixelType::HALF && fast_f16)
			convert_depth(&source_alpha, PixelType::HALF, params, factory, true);
		else if (target_alpha.type == PixelType::FLOAT)
			convert_depth(&source_alpha, PixelType::FLOAT, params, factory, true);
		else if (source_alpha.type == PixelType::BYTE)
			convert_depth(&source_alpha, PixelFormat{ PixelType::WORD, 16, false }, params, factory, true);
		else if (source_alpha.type == PixelType::HALF && (target_alpha.type != PixelType::HALF || !fast_f16))
			convert_depth(&source_alpha, PixelType::FLOAT, params, factory, true);

		// (1.2) Resize plane.
		resize_spec spec{ target_alpha };
		spec.shift_w = source_alpha.active_left;
		spec.shift_h = source_alpha.active_top;
		spec.subwidth = source_alpha.active_width;
		spec.subheight = source_alpha.active_height;

		convert_resize(&source_alpha, spec, params, factory, true);
		zassert_d(!needs_resize(source_alpha, target_alpha), "conversion did not apply");
	}

	// (2) Convert to target pixel format.
	if (needs_depth(source_alpha, target_alpha)) {
		convert_depth(&source_alpha, { target.type, target.depth, target.fullrange }, params, factory, true);
		zassert_d(!needs_depth(source_alpha, target_alpha), "conversion did not apply");
	}
}

GraphBuilder2 &GraphBuilder2::set_source(const state &source) try
{
	if (m_graph)
		error::throw_<error::InternalError>("graph already initialized");

	validate_state(source);

	m_graph = ztd::make_unique<FilterGraph2>();
	m_state = source;

	ImageFilter::image_attributes attr{ source.width, source.height, source.type };
	plane_mask mask{ { true, is_color(source), is_color(source), has_alpha(source) } };

	node_id id = m_graph->add_source(attr, source.subsample_w, source.subsample_h, mask);
	m_plane_ids[PLANE_Y] = id;
	m_plane_ids[PLANE_U] = is_color(source) ? id : -1;
	m_plane_ids[PLANE_V] = is_color(source) ? id : -1;
	m_plane_ids[PLANE_A] = has_alpha(source) ? id : -1;

	return *this;
} catch (const std::bad_alloc &) {
	error::throw_<error::OutOfMemory>();
}

GraphBuilder2 &GraphBuilder2::connect_graph(const state &target, const params *params, FilterFactory2 *factory) try
{
	if (!m_graph)
		error::throw_<error::InternalError>("graph not initialized");

	validate_state(target);

	if (target.active_left != 0 || target.active_top != 0 || target.active_width != target.width || target.active_height != target.height)
		error::throw_<error::ResamplingNotAvailable>("active subregions not supported on target image");
	if (m_state.parity != target.parity)
		error::throw_<error::NoFieldParityConversion>("conversion between field parity not supported");

	DefaultFilterFactory2 default_factory;
	if (!factory)
		factory = &default_factory;

	if (params && cpu_requires_64b_alignment(params->cpu))
		m_graph->set_requires_64b_alignment();

	if (m_state.alpha == AlphaType::STRAIGHT && (needs_colorspace(m_state, target) || needs_resize(m_state, target) || !has_alpha(target)))
		premultiply(params, factory);

	if (!has_alpha(target)) {
		m_plane_ids[PLANE_A] = -1;
		m_state.alpha = AlphaType::NONE;
	}

	if (m_state.alpha == AlphaType::PREMULTIPLED && target.alpha == AlphaType::STRAIGHT) {
		state tmp = target;
		tmp.type = PixelType::FLOAT;
		tmp.subsample_w = 0;
		tmp.subsample_h = 0;
		tmp.alpha = AlphaType::PREMULTIPLED;

		state orig = m_state;
		connect_color_channels(tmp, params, factory);
		connect_alpha_channel(orig, tmp, params, factory);
		unpremultiply(params, factory);

		orig = m_state;
		connect_color_channels(target, params, factory);
		connect_alpha_channel(orig, target, params, factory);
	} else {
		state orig = m_state;
		connect_color_channels(target, params, factory);
		if (has_alpha(m_state))
			connect_alpha_channel(orig, target, params, factory);
	}

	if (!has_alpha(m_state) && has_alpha(target)) {
		add_opaque_alpha();
		m_state.alpha = target.alpha;
	}

	return *this;
} catch (const std::bad_alloc &) {
	error::throw_<error::OutOfMemory>();
}

std::unique_ptr<FilterGraph2> GraphBuilder2::complete_graph() try
{
	if (!m_graph)
		error::throw_<error::InternalError>("graph not initialized");

	id_map map;
	std::copy(std::begin(m_plane_ids), std::end(m_plane_ids), map.begin());

	m_graph->set_output(map);
	return std::move(m_graph);
} catch (const std::bad_alloc &) {
	error::throw_<error::OutOfMemory>();
}

} // namespace graph
} // namespace zimg
