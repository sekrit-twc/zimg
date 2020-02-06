#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/static_map.h"
#include "common/zassert.h"
#include "graph/filtergraph.h"
#include "graph/image_filter.h"
#include "depth/depth.h"

#include "frame.h"
#include "mmap.h"
#include "win32_bitmap.h"

namespace {

bool is_null_device(const std::string &s) noexcept
{
#ifdef _WIN32
	return s == "NUL";
#else
	return s == "/dev/null";
#endif
}

enum class PackingFormat {
	PACK_PLANAR,
	PACK_YUY2,
	PACK_BMP
};

struct PathSpecifier {
	zimg::PixelType type;
	unsigned planes;
	unsigned subsample_w;
	unsigned subsample_h;
	unsigned plane_order[3];
	bool is_yuv;
	PackingFormat packing;
	std::string path;
};

PathSpecifier translate_pathspec_format(const char *format)
{
#define ORDER_NUL { 0, 0, 0 }
#define ORDER_DEF { 0, 1, 2 }
#define ORDER_YVU { 0, 2, 1 }
#define ORDER_GBR { 2, 0, 1 }
#define Z(x) zimg::PixelType::x
	static const zimg::static_string_map<PathSpecifier, 30> map{
		{ "bmp",   { Z(BYTE), 3, 0, 0, ORDER_NUL, false, PackingFormat::PACK_BMP } },
		{ "grey",  { Z(BYTE), 1, 0, 0, ORDER_NUL, true } },
		{ "yuy2",  { Z(BYTE), 3, 1, 0, ORDER_NUL, true,  PackingFormat::PACK_YUY2 } },
		{ "yv12",  { Z(BYTE), 3, 1, 1, ORDER_YVU, true } },
		{ "yv16",  { Z(BYTE), 3, 1, 0, ORDER_YVU, true } },
		{ "yv24",  { Z(BYTE), 3, 0, 0, ORDER_YVU, true } },
		{ "i420",  { Z(BYTE), 3, 1, 1, ORDER_DEF, true } },
		{ "i422",  { Z(BYTE), 3, 1, 0, ORDER_DEF, true } },
		{ "i444",  { Z(BYTE), 3, 0, 0, ORDER_DEF, true } },
		{ "rgbp",  { Z(BYTE), 3, 0, 0, ORDER_DEF, false } },
		{ "gbrp",  { Z(BYTE), 3, 0, 0, ORDER_GBR, false } },

		{ "greyw", { Z(WORD), 1, 0, 0, ORDER_NUL, false } },
		{ "yv12w", { Z(WORD), 3, 1, 1, ORDER_YVU, true } },
		{ "yv16w", { Z(WORD), 3, 1, 0, ORDER_YVU, true } },
		{ "yv24w", { Z(WORD), 3, 0, 0, ORDER_YVU, true } },
		{ "i420w", { Z(WORD), 3, 1, 1, ORDER_DEF, true } },
		{ "i422w", { Z(WORD), 3, 1, 0, ORDER_DEF, true } },
		{ "i444w", { Z(WORD), 3, 0, 0, ORDER_DEF, true } },
		{ "rgbpw", { Z(WORD), 3, 0, 0, ORDER_DEF, false } },
		{ "gbrpw", { Z(WORD), 3, 0, 0, ORDER_GBR, false } },

		{ "greyh", { Z(HALF), 1, 0, 0, ORDER_NUL, false } },
		{ "i420h", { Z(HALF), 3, 1, 1, ORDER_DEF, true } },
		{ "i422h", { Z(HALF), 3, 1, 0, ORDER_DEF, true } },
		{ "i444h", { Z(HALF), 3, 0, 0, ORDER_DEF, true } },
		{ "rgbph", { Z(HALF), 3, 0, 0, ORDER_DEF, false } },

		{ "greys", { Z(FLOAT), 1, 0, 0, ORDER_NUL, false } },
		{ "i420s", { Z(FLOAT), 3, 1, 1, ORDER_DEF, true } },
		{ "i422s", { Z(FLOAT), 3, 1, 0, ORDER_DEF, true } },
		{ "i444s", { Z(FLOAT), 3, 0, 0, ORDER_DEF, true } },
		{ "rgbps", { Z(FLOAT), 3, 0, 0, ORDER_DEF, false } },
	};
#undef ORDER_NUL
#undef ORDER_DEF
#undef ORDER_YVU
#undef ORDER_GBR
#undef Z
	return map[format];
}

PathSpecifier parse_path_specifier(const char *spec, const char *assumed)
{
	PathSpecifier parsed_spec;

	std::string s{ spec };
	std::string format;
	std::string path;

	size_t delimiter_pos = s.find('@');

	if (delimiter_pos == std::string::npos) {
		format = assumed;
		path = s;
	} else {
		format = s.substr(0, delimiter_pos);
		path = s.substr(delimiter_pos + 1);
	}

	parsed_spec = translate_pathspec_format(format.c_str());
	parsed_spec.path = std::move(path);

	return parsed_spec;
}

constexpr ptrdiff_t width_to_stride(unsigned width, zimg::PixelType pixel) noexcept
{
	return zimg::ceil_n(width * zimg::pixel_size(pixel), zimg::ALIGNMENT);
}

} // namespace


ImageFrame::ImageFrame(unsigned width, unsigned height, zimg::PixelType pixel, unsigned planes,
                       bool yuv, unsigned subsample_w, unsigned subsample_h) :
	m_offset{ 0, 832, 2368, 320 },
	m_width{ width },
	m_height{ height },
	m_pixel{ pixel },
	m_planes{ planes },
	m_subsample_w{ subsample_w },
	m_subsample_h{ subsample_h },
	m_yuv{ yuv }
{
	for (unsigned p = 0; p < planes; ++p) {
		unsigned width_p = this->width(p);
		unsigned height_p = this->height(p);
		size_t rowsize_p = width_to_stride(width_p, pixel);

		m_vector[p].resize(rowsize_p * height_p + m_offset[p]);
	}
}

unsigned ImageFrame::width(unsigned plane) const noexcept
{
	return m_width >> ((plane == 1 || plane == 2) ? m_subsample_w : 0);
}

unsigned ImageFrame::height(unsigned plane) const noexcept
{
	return m_height >> ((plane == 1 || plane == 2) ? m_subsample_h : 0);
}

zimg::PixelType ImageFrame::pixel_type() const noexcept { return m_pixel; }
unsigned ImageFrame::planes() const noexcept { return m_planes; }
unsigned ImageFrame::subsample_w() const noexcept { return m_subsample_w; }
unsigned ImageFrame::subsample_h() const noexcept { return m_subsample_h; }
bool ImageFrame::is_yuv() const noexcept { return m_yuv; }

zimg::graph::ImageBuffer<const void> ImageFrame::as_read_buffer(unsigned plane) const noexcept
{
	return const_cast<ImageFrame *>(this)->as_write_buffer(plane);
}

zimg::graph::ColorImageBuffer<const void> ImageFrame::as_read_buffer() const noexcept
{
	zimg::graph::ColorImageBuffer<const void> ret{};

	for (unsigned p = 0; p < std::min(m_planes, 4U); ++p) {
		ret[p] = as_read_buffer(p);
	}
	return ret;
}

zimg::graph::ImageBuffer<void> ImageFrame::as_write_buffer(unsigned plane) noexcept
{
	zassert(plane < m_planes, "plane index out of bounds");
	return{ m_vector[plane].data() + m_offset[plane], width_to_stride(width(plane), m_pixel), zimg::graph::BUFFER_MAX };
}

zimg::graph::ColorImageBuffer<void> ImageFrame::as_write_buffer() noexcept
{
	zimg::graph::ColorImageBuffer<void> ret{};

	for (unsigned p = 0; p < std::min(m_planes, 4U); ++p) {
		ret[p] = as_write_buffer(p);
	}
	return ret;
}


namespace imageframe {

namespace {

struct MappedImageFile {
	std::unique_ptr<MemoryMappedFile> m_handle;
	unsigned m_linewidth[3];
	unsigned m_height[3];
	unsigned m_plane_order[3];
	void *m_ptr[3];
public:
	MappedImageFile(const PathSpecifier &spec, unsigned width, unsigned height, bool write) :
		m_linewidth{},
		m_height{},
		m_plane_order{ spec.plane_order[0], spec.plane_order[1], spec.plane_order[2] },
		m_ptr{}
	{
		size_t size = 0;
		char *ptr;

		zassert_d(spec.planes <= 3, "too many planes");

		for (unsigned p = 0; p < spec.planes; ++p) {
			m_linewidth[p] = (width * zimg::pixel_size(spec.type)) >> ((p == 1 || p == 2) ? spec.subsample_w : 0);
			m_height[p] = height >> ((p == 1 || p == 2) ? spec.subsample_h : 0);
			size += static_cast<size_t>(m_linewidth[p]) * m_height[p];
		}

		if (write)
			m_handle = ztd::make_unique<MemoryMappedFile>(spec.path.c_str(), size, MemoryMappedFile::CREATE_TAG);
		else
			m_handle = ztd::make_unique<MemoryMappedFile>(spec.path.c_str(), MemoryMappedFile::READ_TAG);

		if (m_handle->size() != size)
			throw std::runtime_error{ "bad file size" };

		ptr = static_cast<char *>(const_cast<void *>(m_handle->read_ptr()));
;		for (unsigned p = 0; p < 3; ++p) {
			size_t size_p = static_cast<size_t>(m_linewidth[p]) * m_height[p];

			m_ptr[p] = ptr;
			ptr += size_p;
		}
	}

	zimg::graph::ColorImageBuffer<const void> as_read_buffer() const
	{
		return const_cast<MappedImageFile *>(this)->as_write_buffer();
	}

	zimg::graph::ColorImageBuffer<void> as_write_buffer()
	{
		return{
			{ m_ptr[m_plane_order[0]], static_cast<ptrdiff_t>(m_linewidth[0]), zimg::graph::BUFFER_MAX },
			{ m_ptr[m_plane_order[1]], static_cast<ptrdiff_t>(m_linewidth[1]), zimg::graph::BUFFER_MAX },
			{ m_ptr[m_plane_order[2]], static_cast<ptrdiff_t>(m_linewidth[2]), zimg::graph::BUFFER_MAX },
		};
	}
};

std::unique_ptr<zimg::graph::FilterGraph> setup_read_graph(const PathSpecifier &spec, unsigned width, unsigned height, zimg::PixelType type, bool fullrange)
{
	bool color = spec.planes >= 3;

	auto graph = ztd::make_unique<zimg::graph::FilterGraph>();
	zimg::graph::node_id id_y = graph->add_source({ width, height, type }, color ? spec.subsample_w : 0, color ? spec.subsample_h : 0, { true, color, color, false });
	zimg::graph::node_id id_u = color ? id_y : -1;
	zimg::graph::node_id id_v = color ? id_y : -1;

	if (type != spec.type) {
		zimg::PixelFormat src_format{ spec.type };
		zimg::PixelFormat dst_format{ type };
		src_format.fullrange = fullrange;
		dst_format.fullrange = fullrange;

		auto conv = zimg::depth::DepthConversion{ width, height }
			.set_pixel_in(src_format)
			.set_pixel_out(dst_format);

		id_y = graph->attach_filter(conv.create(), { id_y, -1, -1, -1 }, { true, false, false, false });

		if (color) {
			src_format.chroma = spec.is_yuv;
			dst_format.chroma = spec.is_yuv;

			conv = zimg::depth::DepthConversion{ width >> spec.subsample_w, height >> spec.subsample_h }
				.set_pixel_in(src_format)
				.set_pixel_out(dst_format);
			std::shared_ptr<zimg::graph::ImageFilter> f{ conv.create() };

			id_u = graph->attach_filter(f, { -1, id_u, -1, -1 }, { false, true, false, false });
			id_v = graph->attach_filter(f, { -1, -1, id_v, -1 }, { false, false, true, false });
		}
	}

	graph->set_output({ id_y, id_u, id_v, -1 });
	return graph;
}

ImageFrame read_from_planar(const PathSpecifier &spec, unsigned width, unsigned height, zimg::PixelType type, bool fullrange)
{
	auto graph = setup_read_graph(spec, width, height, type, fullrange);
	zimg::AlignedVector<char> tmp(graph->get_tmp_size());

	MappedImageFile mapped_image{ spec, width, height, false };
	ImageFrame out_image{ width, height, type, spec.planes, spec.is_yuv, spec.subsample_w, spec.subsample_h };

	graph->process(mapped_image.as_read_buffer(), out_image.as_write_buffer(), tmp.data(), nullptr, nullptr);

	return out_image;
}

ImageFrame read_from_bmp(const PathSpecifier &spec, zimg::PixelType type, bool fullrange)
{
	WindowsBitmap bmp_image{ spec.path.c_str(), WindowsBitmap::READ_TAG };
	ImageFrame out_image{ static_cast<unsigned>(bmp_image.width()), static_cast<unsigned>(bmp_image.height()), type, 3 };

	auto graph = setup_read_graph(spec, bmp_image.width(), bmp_image.height(), type, fullrange);
	zimg::AlignedVector<char> tmp(graph->get_tmp_size());

	zimg::graph::ImageBuffer<void> line_buffer[3];
	zimg::AlignedVector<char> planar_tmp(bmp_image.width() * (bmp_image.bit_count() / 8));

	for (unsigned p = 0; p < 3; ++p) {
		void *ptr = planar_tmp.data() + static_cast<size_t>(bmp_image.width()) * p;
		line_buffer[p] = zimg::graph::ImageBuffer<void>{ ptr, bmp_image.width(), 0 };
	}

	struct callback_context_type {
		const WindowsBitmap *bmp;
		const zimg::graph::ImageBuffer<void> *buffer;
	} callback_context = { &bmp_image, line_buffer };

	auto cb = [](void *user, unsigned i, unsigned left, unsigned right) -> int
	{
		callback_context_type *cb_ctx = static_cast<callback_context_type *>(user);
		const zimg::graph::ImageBuffer<uint8_t> *buffer = zimg::graph::static_buffer_cast<uint8_t>(cb_ctx->buffer);
		const WindowsBitmap *bmp = cb_ctx->bmp;

		const uint8_t *base = bmp->read_ptr() + static_cast<ptrdiff_t>(i) * bmp->stride();
		unsigned step = bmp->bit_count() / 8;

		uint8_t *dst_ptr[3] = { buffer[0].data(), buffer[1].data(), buffer[2].data() };

		for (unsigned j = left; j < right; ++j) {
			dst_ptr[0][j] = base[j * step + 2];
			dst_ptr[1][j] = base[j * step + 1];
			dst_ptr[2][j] = base[j * step + 0];
		}

		return 0;
	};

	graph->process(zimg::graph::static_buffer_cast<const void>(line_buffer),
	               out_image.as_write_buffer(),
	               tmp.data(),
	               { cb, &callback_context },
	               nullptr);

	return out_image;
}

ImageFrame read_from_yuy2(const PathSpecifier &spec, unsigned width, unsigned height, zimg::PixelType type, bool fullrange)
{
	MemoryMappedFile mmap_image{ spec.path.c_str(), MemoryMappedFile::READ_TAG };
	ImageFrame out_image{ width, height, type, 3, true, 1, 0 };

	unsigned mmap_linesize = width * 2;

	if (mmap_image.size() != static_cast<size_t>(mmap_linesize) * height)
		throw std::runtime_error{ "bad image size" };

	auto graph = setup_read_graph(spec, width, height, type, fullrange);
	zimg::AlignedVector<char> tmp(graph->get_tmp_size());

	zimg::graph::ImageBuffer<void> line_buffer[3];
	zimg::AlignedVector<char> planar_tmp(mmap_linesize);

	line_buffer[0] = { planar_tmp.data(), static_cast<ptrdiff_t>(width), 0 };
	line_buffer[1] = { planar_tmp.data() + width, static_cast<ptrdiff_t>(width) / 2, 0 };
	line_buffer[2] = { planar_tmp.data() + width + width / 2, static_cast<ptrdiff_t>(width) / 2, 0 };

	struct callback_context_type {
		const void *src_base;
		unsigned linesize;
		const zimg::graph::ImageBuffer<void> *buffer;
	} callback_context = { mmap_image.read_ptr(), mmap_linesize, line_buffer };

	auto cb = [](void *user, unsigned i, unsigned left, unsigned right) -> int
	{
		callback_context_type *cb_ctx = static_cast<callback_context_type *>(user);
		const zimg::graph::ImageBuffer<uint8_t> *buffer = zimg::graph::static_buffer_cast<uint8_t>(cb_ctx->buffer);

		left = left % 2 ? left - 1 : left;
		right = right % 2 ? right + 1 : right;

		const uint8_t *base = static_cast<const uint8_t *>(cb_ctx->src_base) + static_cast<size_t>(i) * cb_ctx->linesize;
		uint8_t *dst_ptr[3] = { buffer[0].data(), buffer[1].data(), buffer[2].data() };

		for (unsigned j = left; j < right; j += 2) {
			dst_ptr[0][j + 0] = base[j * 2 + 0];
			dst_ptr[0][j + 1] = base[j * 2 + 2];
			dst_ptr[1][j / 2] = base[j * 2 + 1];
			dst_ptr[2][j / 2] = base[j * 2 + 3];
		}

		return 0;
	};

	graph->process(zimg::graph::static_buffer_cast<const void>(line_buffer),
	               out_image.as_write_buffer(),
	               tmp.data(),
	               { cb, &callback_context },
	               nullptr);

	return out_image;
}

ImageFrame read_from_pathspec(const PathSpecifier &spec, unsigned width, unsigned height, zimg::PixelType type, bool fullrange)
{
	switch (spec.packing) {
	case PackingFormat::PACK_PLANAR:
		return read_from_planar(spec, width, height, type, fullrange);
	case PackingFormat::PACK_BMP:
		return read_from_bmp(spec, type, fullrange);
	case PackingFormat::PACK_YUY2:
		return read_from_yuy2(spec, width, height, type, fullrange);
	default:
		zassert(false, "bad packing type");
		throw std::runtime_error{ "" };
	}
}


std::unique_ptr<zimg::graph::FilterGraph> setup_write_graph(const PathSpecifier &spec, unsigned width, unsigned height, zimg::PixelType type,
                                                            unsigned depth_in, bool fullrange)
{
	bool color = spec.planes >= 3;

	auto graph = ztd::make_unique<zimg::graph::FilterGraph>();
	zimg::graph::node_id id_y = graph->add_source({ width, height, type }, color ? spec.subsample_w : 0, color ? spec.subsample_h : 0, { true, color, color, false });
	zimg::graph::node_id id_u = color ? id_y : -1;
	zimg::graph::node_id id_v = color ? id_y : -1;

	if (type != spec.type || depth_in != zimg::pixel_depth(type)) {
		zimg::PixelFormat src_format{ type, depth_in };
		zimg::PixelFormat dst_format{ spec.type };
		src_format.fullrange = fullrange;
		dst_format.fullrange = fullrange;

		auto conv = zimg::depth::DepthConversion{ width, height }
			.set_pixel_in(src_format)
			.set_pixel_out(dst_format);

		id_y = graph->attach_filter(conv.create(), { id_y, -1, -1, -1 }, { true, false, false, false });

		if (color) {
			src_format.chroma = spec.is_yuv;
			dst_format.chroma = spec.is_yuv;

			auto conv_uv = zimg::depth::DepthConversion{ width >> spec.subsample_w, height >> spec.subsample_h }
				.set_pixel_in(src_format)
				.set_pixel_out(dst_format);
			std::shared_ptr<zimg::graph::ImageFilter> f{ conv_uv.create() };

			id_u = graph->attach_filter(f, { -1, id_u, -1, -1 }, { false, true, false, false });
			id_v = graph->attach_filter(f, { -1, -1, id_v, -1 }, { false, false, true, false });
		}
	}

	graph->set_output({ id_y, id_u, id_v, -1 });
	return graph;
}

void write_to_planar(const ImageFrame &frame, const PathSpecifier &spec, unsigned depth_in, bool fullrange)
{
	auto graph = setup_write_graph(spec, frame.width(), frame.height(), frame.pixel_type(), depth_in, fullrange);
	zimg::AlignedVector<char> tmp(graph->get_tmp_size());

	MappedImageFile mapped_image{ spec, frame.width(), frame.height(), true };
	graph->process(frame.as_read_buffer(), mapped_image.as_write_buffer(), tmp.data(), nullptr, nullptr);
}

void write_to_bmp(const ImageFrame &frame, const PathSpecifier &spec, unsigned depth_in, bool fullrange)
{
	WindowsBitmap bmp_image{ spec.path.c_str(), static_cast<int>(frame.width()), static_cast<int>(frame.height()), static_cast<int>(frame.planes()) * 8 };

	auto graph = setup_write_graph(spec, frame.width(), frame.height(), frame.pixel_type(), depth_in, fullrange);
	zimg::AlignedVector<char> tmp(graph->get_tmp_size());

	zimg::graph::ImageBuffer<void> line_buffer[3];
	zimg::AlignedVector<char> planar_tmp(bmp_image.width() * (bmp_image.bit_count() / 8));

	for (unsigned p = 0; p < 3; ++p) {
		void *ptr = planar_tmp.data() + static_cast<size_t>(bmp_image.width()) * p;
		line_buffer[p] = zimg::graph::ImageBuffer<void>{ ptr, bmp_image.width(), 0 };
	}

	struct callback_context_type {
		WindowsBitmap *bmp;
		const zimg::graph::ImageBuffer<void> *buffer;
	} callback_context = { &bmp_image, line_buffer };

	auto cb = [](void *user, unsigned i, unsigned left, unsigned right) -> int
	{
		callback_context_type *cb_ctx = static_cast<callback_context_type *>(user);
		WindowsBitmap *bmp = cb_ctx->bmp;
		const zimg::graph::ImageBuffer<const uint8_t> *buffer = zimg::graph::static_buffer_cast<const uint8_t>(cb_ctx->buffer);

		uint8_t *base = bmp->write_ptr() + static_cast<ptrdiff_t>(i) * bmp->stride();
		unsigned step = bmp->bit_count() / 8;

		const uint8_t *src_ptr[3] = { buffer[0].data(), buffer[1].data(), buffer[2].data() };

		for (unsigned j = left; j < right; ++j) {
			base[j * step + 0] = src_ptr[2][j];
			base[j * step + 1] = src_ptr[1][j];
			base[j * step + 2] = src_ptr[0][j];
		}

		return 0;
	};

	graph->process(frame.as_read_buffer(), line_buffer, tmp.data(), nullptr, { cb, &callback_context });
}

void write_to_yuy2(const ImageFrame &frame, const PathSpecifier &spec, unsigned depth_in, bool fullrange)
{
	unsigned mmap_linesize = frame.width() * 2;

	MemoryMappedFile mmap_image{ spec.path.c_str(), static_cast<size_t>(mmap_linesize) * frame.height(), MemoryMappedFile::CREATE_TAG };

	auto graph = setup_write_graph(spec, frame.width(), frame.height(), frame.pixel_type(), depth_in, fullrange);
	zimg::AlignedVector<char> tmp(graph->get_tmp_size());

	zimg::graph::ImageBuffer<void> line_buffer[3];
	zimg::AlignedVector<char> planar_tmp(mmap_linesize);

	line_buffer[0] = { planar_tmp.data(), static_cast<ptrdiff_t>(frame.width()), 0 };
	line_buffer[1] = { planar_tmp.data() + frame.width(), static_cast<ptrdiff_t>(frame.width()) / 2, 0 };
	line_buffer[2] = { planar_tmp.data() + frame.width() + frame.width() / 2, static_cast<ptrdiff_t>(frame.width()) / 2, 0 };

	struct callback_context_type {
		void *dst_base;
		unsigned linesize;
		const zimg::graph::ImageBuffer<void> *buffer;
	} callback_context = { mmap_image.write_ptr(), mmap_linesize, line_buffer };

	auto cb = [](void *user, unsigned i, unsigned left, unsigned right) -> int
	{
		callback_context_type *cb_ctx = static_cast<callback_context_type *>(user);
		const zimg::graph::ImageBuffer<const uint8_t> *buffer = zimg::graph::static_buffer_cast<const uint8_t>(cb_ctx->buffer);

		left = left % 2 ? left - 1 : left;
		right = right % 2 ? right + 1 : right;

		uint8_t *base = static_cast<uint8_t *>(cb_ctx->dst_base) + static_cast<size_t>(i) * cb_ctx->linesize;
		const uint8_t *src_ptr[3] = { buffer[0].data(), buffer[1].data(), buffer[2].data() };

		for (unsigned j = left; j < right; j += 2) {
			base[j * 2 + 0] = src_ptr[0][j + 0];
			base[j * 2 + 1] = src_ptr[1][j / 2];
			base[j * 2 + 2] = src_ptr[0][j + 1];
			base[j * 2 + 3] = src_ptr[2][j / 2];
		}

		return 0;
	};

	graph->process(frame.as_read_buffer(), line_buffer, tmp.data(), nullptr, { cb, &callback_context });
}

} // namespace


ImageFrame read(const char *pathspec, const char *assumed, unsigned width, unsigned height)
{
	PathSpecifier spec = parse_path_specifier(pathspec, assumed);
	return read_from_pathspec(spec, width, height, spec.type, false);
}

ImageFrame read(const char *pathspec, const char *assumed, unsigned width, unsigned height, zimg::PixelType type, bool fullrange)
{
	return read_from_pathspec(parse_path_specifier(pathspec, assumed), width, height, type, fullrange);
}

void write(const ImageFrame &frame, const char *pathspec, const char *assumed, bool fullrange)
{
	write(frame, pathspec, assumed, zimg::pixel_depth(frame.pixel_type()), fullrange);
}

void write(const ImageFrame &frame, const char *pathspec, const char *assumed, unsigned depth_in, bool fullrange)
{
	PathSpecifier spec = parse_path_specifier(pathspec, assumed);

	if (is_null_device(spec.path))
		return;

	if (spec.planes != frame.planes())
		throw std::logic_error{ "incompatible plane count in format" };
	if (spec.subsample_w != frame.subsample_w() || spec.subsample_h != frame.subsample_h())
		throw std::logic_error{ "incompatible subsampling in format" };

	switch (spec.packing) {
	case PackingFormat::PACK_PLANAR:
		write_to_planar(frame, spec, depth_in, fullrange);
		return;
	case PackingFormat::PACK_BMP:
		write_to_bmp(frame, spec, depth_in, fullrange);
		return;
	case PackingFormat::PACK_YUY2:
		write_to_yuy2(frame, spec, depth_in, fullrange);
		return;
	}
}

} // namespace imageframe
