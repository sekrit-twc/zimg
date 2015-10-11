#include <iterator>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "resize/filter.h"
#include "filtergraph.h"
#include "basic_filter.h"
#include "graphbuilder.h"
#include "image_filter.h"

namespace zimg {;
namespace graph {;

namespace {;

double chroma_shift_raw(GraphBuilder::ChromaLocationW loc, GraphBuilder::FieldParity)
{
	if (loc == GraphBuilder::ChromaLocationW::CHROMA_W_LEFT)
		return -0.5;
	else
		return 0.0;
}

double chroma_shift_raw(GraphBuilder::ChromaLocationH loc, GraphBuilder::FieldParity parity)
{
	double shift;

	if (loc == GraphBuilder::ChromaLocationH::CHROMA_H_TOP)
		shift = -0.5;
	else if (loc == GraphBuilder::ChromaLocationH::CHROMA_H_BOTTOM)
		shift = 0.5;
	else
		shift = 0;

	if (parity == GraphBuilder::FieldParity::FIELD_TOP)
		shift = (shift - 0.5) / 2.0;
	else if (parity == GraphBuilder::FieldParity::FIELD_BOTTOM)
		shift = (shift + 0.5) / 2.0;

	return shift;
}

template <class T>
double chroma_shift_factor(T loc_in, T loc_out, unsigned subsample_in, unsigned subsample_out, GraphBuilder::FieldParity parity, unsigned src_dim, unsigned dst_dim)
{
	double shift = 0.0;
	double sub_scale = 1.0 / (1 << subsample_in);

	if (subsample_in)
		shift -= sub_scale * chroma_shift_raw(loc_in, parity);
	if (subsample_out)
		shift += sub_scale * chroma_shift_raw(loc_out, parity) * (double)src_dim / dst_dim;

	return shift;
}

double luma_shift_factor(GraphBuilder::FieldParity parity, unsigned src_height, unsigned dst_height)
{
	double shift = 0.0;

	if (parity == GraphBuilder::FieldParity::FIELD_TOP)
		shift = -0.25;
	else if (parity == GraphBuilder::FieldParity::FIELD_BOTTOM)
		shift = 0.25;

	return shift * (double)src_height / dst_height - shift;
}


bool is_greyscale(const GraphBuilder::state &state)
{
	return state.color == GraphBuilder::ColorFamily::COLOR_GREY;
}

bool is_rgb(const GraphBuilder::state &state)
{
	return state.color == GraphBuilder::ColorFamily::COLOR_RGB;
}

bool is_yuv(const GraphBuilder::state &state)
{
	return state.color == GraphBuilder::ColorFamily::COLOR_YUV;
}

void validate_state(const GraphBuilder::state &state)
{
	if (!state.width || !state.height)
		throw error::ZeroImageSize{ "image dimensions must be non-zero" };

	if (is_greyscale(state)) {
		if (state.subsample_w || state.subsample_h)
			throw error::GreyscaleSubsampling{ "cannot subsample greyscale image" };
		if (state.colorspace.matrix == zimg::colorspace::MatrixCoefficients::MATRIX_RGB)
			throw error::ColorFamilyMismatch{ "GREY color family cannot be RGB" };
	}

	if (is_rgb(state)) {
		if (state.subsample_w || state.subsample_h)
			throw zimg::error::UnsupportedSubsampling{ "subsampled RGB image not supported" };
		if (state.colorspace.matrix != zimg::colorspace::MatrixCoefficients::MATRIX_UNSPECIFIED &&
			state.colorspace.matrix != zimg::colorspace::MatrixCoefficients::MATRIX_RGB)
			throw error::ColorFamilyMismatch{ "RGB color family cannot be YUV" };
	}

	if (is_yuv(state)) {
		if (state.colorspace.matrix == zimg::colorspace::MatrixCoefficients::MATRIX_RGB)
			throw error::ColorFamilyMismatch{ "YUV color family cannot be RGB" };
	}

	if (state.subsample_h > 1 && state.parity != GraphBuilder::FieldParity::FIELD_PROGRESSIVE)
		throw error::UnsupportedSubsampling{ "vertical subsampling greater than 2x is not supported" };
	if (state.subsample_w > 2 || state.subsample_h > 2)
		throw error::UnsupportedSubsampling{ "subsampling greater than 4x is not supported" };

	if (state.width % (1 << state.subsample_w) || state.height % (1 << state.subsample_h))
		throw error::ImageNotDivislbe{ "image dimensions must be divisible by subsampling factor" };

	if (state.depth > pixel_depth(state.type))
		throw error::BitDepthOverflow{ "bit depth exceeds limits of type" };
	if (!state.fullrange && state.depth < 8)
		throw error::BitDepthOverflow{ "bit depth must be at least 8 for limited range" };
}

bool needs_colorspace(const GraphBuilder::state &source, const GraphBuilder::state &target)
{
	colorspace::ColorspaceDefinition csp_in = source.colorspace;
	colorspace::ColorspaceDefinition csp_out = target.colorspace;

	if (is_greyscale(source))
		csp_in.matrix = csp_out.matrix;

	return csp_in != csp_out;
}

bool needs_depth(const GraphBuilder::state &source, const GraphBuilder::state &target)
{
	return source.type != target.type || source.depth != target.depth || source.fullrange != target.fullrange;
}

bool needs_resize(const GraphBuilder::state &source, const GraphBuilder::state &target)
{
	if (is_greyscale(source) || is_greyscale(target))
		return source.width != target.width ||
		       source.height != target.height;
	else
		return source.width != target.width ||
	           source.height != target.height ||
	           source.subsample_w != target.subsample_w ||
	           source.subsample_h != target.subsample_h ||
	           (source.subsample_w && source.chroma_location_w != target.chroma_location_w) ||
	           (source.subsample_h && source.chroma_location_h != target.chroma_location_h);
}

} // namespace


FilterFactory::~FilterFactory()
{
}

auto DefaultFilterFactory::create_colorspace(const colorspace::ColorspaceConversion &conv) -> filter_list
{
	auto filter = conv.create();
	return{ std::make_move_iterator(&filter), std::make_move_iterator(&filter + 1) };
}

auto DefaultFilterFactory::create_depth(const depth::DepthConversion &conv) -> filter_list
{
	auto filter = conv.create();
	return{ std::make_move_iterator(&filter), std::make_move_iterator(&filter + 1) };
}

auto DefaultFilterFactory::create_resize(const resize::ResizeConversion &conv) -> filter_list
{
	auto filter_pair = conv.create();

	std::unique_ptr<ImageFilter> filters[2] = { std::move(filter_pair.first), std::move(filter_pair.second) };
	return{ std::make_move_iterator(filters), std::make_move_iterator(filters + 2) };
}


GraphBuilder::resize_spec::resize_spec(const state &state) :
	width{ state.width },
	height{ state.height },
	subsample_w{ state.subsample_w },
	subsample_h{ state.subsample_h },
	shift_w{},
	shift_h{},
	subwidth{ (double)state.width },
	subheight{ (double)state.height },
	chroma_location_w{ state.chroma_location_w },
	chroma_location_h{ state.chroma_location_h }
{
}

GraphBuilder::GraphBuilder() :
	m_factory{},
	m_state{}
{
}

GraphBuilder::~GraphBuilder()
{
}

void GraphBuilder::attach_filter(std::unique_ptr<ImageFilter> &&filter)
{
	if (!filter)
		return;

	m_graph->attach_filter(std::move(filter));
}

void GraphBuilder::attach_filter_uv(std::unique_ptr<ImageFilter> &&filter)
{
	if (!filter)
		return;

	m_graph->attach_filter_uv(std::move(filter));
}

void GraphBuilder::color_to_grey(colorspace::MatrixCoefficients matrix)
{
	if (m_state.color == ColorFamily::COLOR_GREY)
		return;
	if (m_state.color == ColorFamily::COLOR_RGB)
		throw error::InternalError{ "cannot discard chroma planes from RGB" };
	if (matrix == colorspace::MatrixCoefficients::MATRIX_RGB)
		throw error::InternalError{ "GREY color family cannot be RGB" };

	m_graph->color_to_grey();
	m_state.color = ColorFamily::COLOR_GREY;
	m_state.colorspace.matrix = matrix;
}

void GraphBuilder::grey_to_color(ColorFamily color, colorspace::MatrixCoefficients matrix, unsigned subsample_w, unsigned subsample_h,
                                 ChromaLocationW chroma_location_w, ChromaLocationH chroma_location_h)
{
	if (color == ColorFamily::COLOR_GREY)
		return;
	if (color == ColorFamily::COLOR_RGB && matrix != colorspace::MatrixCoefficients::MATRIX_UNSPECIFIED && matrix != colorspace::MatrixCoefficients::MATRIX_RGB)
		throw error::ColorFamilyMismatch{ "RGB color family cannot be YUV" };

	if (!subsample_w)
		chroma_location_w = ChromaLocationW::CHROMA_W_CENTER;
	if (!subsample_h)
		chroma_location_h = ChromaLocationH::CHROMA_H_CENTER;

	m_graph->grey_to_color(color == ColorFamily::COLOR_YUV, subsample_w, subsample_h, m_state.depth);

	m_state.subsample_w = subsample_w;
	m_state.subsample_h = subsample_h;
	m_state.color = color;
	m_state.colorspace.matrix = matrix;

	m_state.chroma_location_w = chroma_location_w;
	m_state.chroma_location_h = chroma_location_h;
}

void GraphBuilder::convert_colorspace(const colorspace::ColorspaceDefinition &colorspace, const params *params)
{
	if (!m_factory)
		throw error::InternalError{ "filter factory not set" };
	if (is_greyscale(m_state))
		throw error::NoColorspaceConversion{ "cannot apply colorspace conversion to greyscale image" };
	if (m_state.colorspace == colorspace)
		return;

	CPUClass cpu = params ? params->cpu : zimg::CPUClass::CPU_AUTO;

	auto conv = colorspace::ColorspaceConversion{ m_state.width, m_state.height }.
		set_csp_in(m_state.colorspace).
		set_csp_out(colorspace).
		set_cpu(cpu);

	for (auto &&filter : m_factory->create_colorspace(conv)) {
		attach_filter(std::move(filter));
	}

	m_state.color = (colorspace.matrix == colorspace::MatrixCoefficients::MATRIX_RGB) ? ColorFamily::COLOR_RGB : ColorFamily::COLOR_YUV;
	m_state.colorspace = colorspace;
}

void GraphBuilder::convert_depth(const PixelFormat &format, const params *params)
{
	if (!m_factory)
		throw error::InternalError{ "filter factory not set" };

	PixelFormat src_format{ m_state.type, m_state.depth, m_state.fullrange, false };

	if (src_format == format)
		return;

	CPUClass cpu = params ? params->cpu : zimg::CPUClass::CPU_AUTO;
	depth::DitherType dither_type = params ? params->dither_type : depth::DitherType::DITHER_NONE;

	auto conv = depth::DepthConversion{ m_state.width, m_state.height }.
		set_pixel_in(src_format).
		set_pixel_out(format).
		set_dither_type(dither_type).
		set_cpu(cpu);

	FilterFactory::filter_list filter_list = m_factory->create_depth(conv);
	FilterFactory::filter_list filter_list_uv;

	if (is_yuv(m_state)) {
		conv.width >>= m_state.subsample_w;
		conv.height >>= m_state.subsample_h;
		conv.pixel_in.chroma = true;
		conv.pixel_out.chroma = true;

		filter_list_uv = m_factory->create_depth(conv);
	} else if (is_rgb(m_state)) {
		for (auto &&filter : filter_list) {
			filter = ztd::make_unique<MuxFilter>(std::move(filter));
		}
	}

	for (auto &&filter : filter_list) {
		attach_filter(std::move(filter));
	}
	for (auto &&filter : filter_list_uv) {
		attach_filter_uv(std::move(filter));
	}

	m_state.type = format.type;
	m_state.depth = format.depth;
	m_state.fullrange = format.fullrange;
}

void GraphBuilder::convert_resize(const resize_spec &spec, const params *params)
{
	if (!m_factory)
		throw error::InternalError{ "filter factory not set" };

	resize::BicubicFilter bicubic_filter{ 1.0 / 3.0, 1.0 / 3.0 };
	resize::BilinearFilter bilinear_filter;

	unsigned subsample_w = spec.subsample_w;
	unsigned subsample_h = spec.subsample_h;
	ChromaLocationW chroma_location_w = spec.chroma_location_w;
	ChromaLocationH chroma_location_h = spec.chroma_location_h;

	bool image_shifted = spec.shift_w != 0.0 ||
	                     spec.shift_h != 0.0 ||
	                     m_state.width != spec.subwidth ||
	                     m_state.height != spec.subheight;

	if (is_greyscale(m_state)) {
		subsample_w = 0;
		subsample_h = 0;
	}

	if (!subsample_w)
		chroma_location_w = ChromaLocationW::CHROMA_W_CENTER;
	if (!subsample_h)
		chroma_location_h = ChromaLocationH::CHROMA_H_CENTER;

	if (m_state.width == spec.width &&
	    m_state.height == spec.height &&
	    m_state.subsample_w == subsample_w &&
	    m_state.subsample_h == subsample_h &&
	    m_state.chroma_location_w == chroma_location_w &&
	    m_state.chroma_location_h == chroma_location_h &&
	    !image_shifted)
		return;

	const resize::Filter *resample_filter = params ? params->filter.get() : &bicubic_filter;
	const resize::Filter *resample_filter_uv = params ? params->filter_uv.get() : &bilinear_filter;
	CPUClass cpu = params ? params->cpu : CPUClass::CPU_AUTO;

	bool do_resize_luma = m_state.width != spec.width || m_state.height != spec.height || image_shifted;
	bool do_resize_chroma = (m_state.width >> m_state.subsample_w != spec.width >> subsample_w) ||
	                        (m_state.height >> m_state.subsample_h != spec.height >> subsample_h) ||
	                        ((m_state.subsample_w || subsample_w) && m_state.chroma_location_w != chroma_location_w) ||
	                        ((m_state.subsample_h || subsample_h) && m_state.chroma_location_h != chroma_location_h) ||
	                        image_shifted;

	FilterFactory::filter_list filter_list;
	FilterFactory::filter_list filter_list_uv;

	if (do_resize_luma) {
		double extra_shift_h = luma_shift_factor(m_state.parity, m_state.height, spec.height);

		auto conv = resize::ResizeConversion{ m_state.width, m_state.height, m_state.type }.
			set_depth(m_state.depth).
			set_filter(resample_filter).
			set_dst_width(spec.width).
			set_dst_height(spec.height).
			set_shift_w(spec.shift_w).
			set_shift_h(spec.shift_h + extra_shift_h).
			set_subwidth(spec.subwidth).
			set_subheight(spec.subheight).
			set_cpu(cpu);

		filter_list = m_factory->create_resize(conv);

		if (is_rgb(m_state)) {
			for (auto &&filter : filter_list) {
				filter = ztd::make_unique<MuxFilter>(std::move(filter));
			}
		}
	}

	if (is_yuv(m_state) && do_resize_chroma) {
		double extra_shift_w = chroma_shift_factor(m_state.chroma_location_w, chroma_location_w, m_state.subsample_w, subsample_w,
		                                           m_state.parity, m_state.width, spec.width);
		double extra_shift_h = chroma_shift_factor(m_state.chroma_location_h, chroma_location_h, m_state.subsample_h, subsample_h,
		                                           m_state.parity, m_state.height, spec.height);

		unsigned chroma_width_in = m_state.width >> m_state.subsample_w;
		unsigned chroma_height_in = m_state.height >> m_state.subsample_h;
		unsigned chroma_width_out = spec.width >> subsample_w;
		unsigned chroma_height_out = spec.height >> subsample_h;

		auto conv = resize::ResizeConversion{ chroma_width_in, chroma_height_in, m_state.type }.
			set_depth(m_state.depth).
			set_filter(resample_filter_uv).
			set_dst_width(chroma_width_out).
			set_dst_height(chroma_height_out).
			set_shift_w(spec.shift_w / (1 << m_state.subsample_w) + extra_shift_w).
			set_shift_h(spec.shift_h / (1 << m_state.subsample_h) + extra_shift_h).
			set_subwidth(spec.subwidth / (1 << m_state.subsample_w)).
			set_subheight(spec.subheight / (1 << m_state.subsample_h)).
			set_cpu(cpu);

		filter_list_uv = m_factory->create_resize(conv);
	}

	for (auto &&filter : filter_list) {
		attach_filter(std::move(filter));
	}
	for (auto &&filter : filter_list_uv) {
		attach_filter_uv(std::move(filter));
	}

	m_state.width = spec.width;
	m_state.height = spec.height;
	m_state.subsample_w = subsample_w;
	m_state.subsample_h = subsample_h;
	m_state.chroma_location_w = chroma_location_w;
	m_state.chroma_location_h = chroma_location_h;
}

GraphBuilder &GraphBuilder::set_factory(FilterFactory *factory)
{
	m_factory = factory;
	return *this;
}

GraphBuilder &GraphBuilder::set_source(const state &source)
{
	if (m_graph)
		throw error::InternalError{ "source already set" };

	validate_state(source);
	m_graph = ztd::make_unique<FilterGraph>(source.width, source.height, source.type, source.subsample_w, source.subsample_h, !is_greyscale(source));
	m_state = source;

	return *this;
}

GraphBuilder &GraphBuilder::connect_graph(const state &target, const params *params)
{
	if (!m_graph)
		throw error::InternalError{ "no active graph" };

	validate_state(target);

	if (m_state.parity != target.parity)
		throw error::NoFieldParityConversion{ "conversion between field parity not supported" };

	while (true) {
		if (needs_colorspace(m_state, target)) {
			resize_spec spec{ m_state };
			spec.subsample_w = 0;
			spec.subsample_h = 0;

			if (is_greyscale(m_state))
				grey_to_color(target.color, target.colorspace.matrix, 0, 0, target.chroma_location_w, target.chroma_location_h);

			if (m_state.type != PixelType::FLOAT)
				convert_depth(PixelType::FLOAT, params);

			convert_resize(spec, params);
			convert_colorspace(target.colorspace, params);

			if (is_greyscale(target))
				color_to_grey(target.colorspace.matrix);
		} else if (!is_greyscale(m_state) && is_greyscale(target)) {
			color_to_grey(target.colorspace.matrix);
		} else if (needs_resize(m_state, target)) {
			if (m_state.type == PixelType::BYTE)
				convert_depth(PixelType::WORD, params);
			if (m_state.type == PixelType::HALF)
				convert_depth(PixelType::FLOAT, params);

			resize_spec spec{ m_state };
			spec.width = target.width;
			spec.height = target.height;
			spec.subsample_w = target.subsample_w;
			spec.subsample_h = target.subsample_h;
			spec.chroma_location_w = target.chroma_location_w;
			spec.chroma_location_h = target.chroma_location_h;

			convert_resize(spec, params);
		} else if (needs_depth(m_state, target)) {
			PixelFormat format{ target.type, target.depth, target.fullrange, false };
			convert_depth(format, params);
		} else if (is_greyscale(m_state) && !is_greyscale(target)) {
			grey_to_color(target.color, target.colorspace.matrix, target.subsample_w, target.subsample_h, target.chroma_location_w, target.chroma_location_h);
		} else {
			break;
		}
	}

	return *this;
}

std::unique_ptr<FilterGraph> GraphBuilder::complete_graph()
{
	m_graph->complete();
	return std::move(m_graph);
}

} // namespace graph
} // namespace zimg
