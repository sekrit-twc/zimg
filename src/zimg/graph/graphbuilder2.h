#pragma once

#ifndef ZIMG_GRAPH_GRAPHBUILDER2_H_
#define ZIMG_GRAPH_GRAPHBUILDER2_H_

#include <memory>
#include <vector>
#include "common/pixel.h"
#include "colorspace/colorspace.h"
#include "depth/depth.h"
#include "resize/resize.h"
#include "unresize/unresize.h"

namespace zimg {

enum class CPUClass;

namespace graph {

class FilterGraph2;
class ImageFilter;

/**
 * Factory interface for filter instantiation.
 */
class FilterFactory2 {
public:
	typedef std::vector<std::unique_ptr<ImageFilter>> filter_list;

	/**
	 * Destroy factory.
	 */
	virtual ~FilterFactory2() = default;

	/**
	 * Create filters implementing colorspace conversion.
	 *
	 * @param conv conversion specifier
	 * @return list of filters
	 */
	virtual filter_list create_colorspace(const colorspace::ColorspaceConversion &conv) = 0;

	/**
	 * Create filters implementing depth conversion.
	 *
	 * @see create_colorspace
	 */
	virtual filter_list create_depth(const depth::DepthConversion &conv) = 0;

	/**
	 * Create filters implementing resizing.
	 *
	 * @see create_colorspace
	 */
	virtual filter_list create_resize(const resize::ResizeConversion &conv) = 0;

	/**
	 * Create filters implementing unresizing.
	 *
	 * @see create_unresize
	 */
	virtual filter_list create_unresize(const unresize::UnresizeConversion &conv) = 0;
};

/**
 * Default implementation of factory interface.
 */
class DefaultFilterFactory2 : public FilterFactory2 {
public:
	filter_list create_colorspace(const colorspace::ColorspaceConversion &conv) override;

	filter_list create_depth(const depth::DepthConversion &conv) override;

	filter_list create_resize(const resize::ResizeConversion &conv) override;

	filter_list create_unresize(const unresize::UnresizeConversion &conv) override;
};


/**
 * Manages initialization of filter graphs.
 */
class GraphBuilder2 {
public:
	enum class ColorFamily {
		GREY,
		RGB,
		YUV,
	};

	enum class AlphaType {
		NONE,
		STRAIGHT,
		PREMULTIPLED,
	};

	enum class FieldParity {
		PROGRESSIVE,
		TOP,
		BOTTOM,
	};

	enum class ChromaLocationW {
		LEFT,
		CENTER,
	};

	enum class ChromaLocationH {
		CENTER,
		TOP,
		BOTTOM,
	};

	/**
	 * Filter instantiation parameters.
	 */
	struct params {
		std::unique_ptr<const resize::Filter> filter;
		std::unique_ptr<const resize::Filter> filter_uv;
		bool unresize;
		depth::DitherType dither_type;
		double peak_luminance;
		bool approximate_gamma;
		bool scene_referred;
		CPUClass cpu;

		params() noexcept;
	};

	/**
	 * Image format specifier.
	 */
	struct state {
		unsigned width;
		unsigned height;
		PixelType type;
		unsigned subsample_w;
		unsigned subsample_h;

		ColorFamily color;
		colorspace::ColorspaceDefinition colorspace;

		unsigned depth;
		bool fullrange;

		FieldParity parity;
		ChromaLocationW chroma_location_w;
		ChromaLocationH chroma_location_h;

		double active_left;
		double active_top;
		double active_width;
		double active_height;

		AlphaType alpha;
	};
private:
	struct resize_spec;

	std::unique_ptr<FilterGraph2> m_graph;
	state m_state;
	int m_plane_ids[4];

	static state make_alpha_state(const state &s);

	void attach_greyscale_filter(std::shared_ptr<ImageFilter> filter, int plane, bool dep = true);

	void attach_color_filter(std::shared_ptr<ImageFilter> filter);

	void convert_colorspace(const colorspace::ColorspaceDefinition &colorspace, const params *params, FilterFactory2 *factory);

	void convert_depth(state *state, const PixelFormat &format, const params *params, FilterFactory2 *factory, bool alpha);

	void convert_resize(state *state, const resize_spec &spec, const params *params, FilterFactory2 *factory, bool alpha);

	void connect_color_channels(const state &target, const params *params, FilterFactory2 *factory);

	void connect_alpha_channel(const state &orig, const state &target, const params *params, FilterFactory2 *factory);

	void add_opaque_alpha();

	void discard_chroma();

	void grey_to_color(ColorFamily color, unsigned subsample_w, unsigned subsample_h, ChromaLocationW chroma_loc_w, ChromaLocationH chroma_loc_h);

	void premultiply(const params *params, FilterFactory2 *factory);

	void unpremultiply(const params *params, FilterFactory2 *factory);
public:
	/**
	 * Default construct GraphBuilder, creating an empty graph.
	 */
	GraphBuilder2() noexcept;

	/**
	 * Destroy builder.
	 */
	~GraphBuilder2();

	/**
	 * Set image format of graph input.
	 *
	 * @param source image format
	 * @return reference to self
	 */
	GraphBuilder2 &set_source(const state &source);

	/**
	 * Connect graph to target image format.
	 *
	 * @param target image format
	 * @param params filter creation parameters
	 * @return reference to self
	 */
	GraphBuilder2 &connect_graph(const state &target, const params *params, FilterFactory2 *factory = nullptr);

	/**
	 * Finalize and return managed graph.
	 *
	 * @return graph
	 */
	std::unique_ptr<FilterGraph2> complete_graph();
};


} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_GRAPHBUILDER2_H_
