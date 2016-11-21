#pragma once

#ifndef WIN32_BITMAP_H_
#define WIN32_BITMAP_H_

#include <cstddef>
#include <memory>
#include <stdexcept>

struct BitmapDataError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class WindowsBitmap {
	class impl;

	struct read_tag {};
	struct write_tag {};
public:
	static const read_tag READ_TAG;
	static const write_tag WRITE_TAG;
private:
	std::unique_ptr<impl> m_impl;

	impl *get_impl() noexcept { return m_impl.get(); }
	const impl *get_impl() const noexcept { return m_impl.get(); }
public:
	WindowsBitmap(WindowsBitmap &&other) noexcept;

	WindowsBitmap(const char *path, read_tag);

	WindowsBitmap(const char *path, write_tag);

	WindowsBitmap(const char *path, int width, int height, int bit_count);

	~WindowsBitmap();

	WindowsBitmap &operator=(WindowsBitmap &&other) noexcept;

	ptrdiff_t stride() const noexcept;

	int width() const noexcept;

	int height() const noexcept;

	int bit_count() const noexcept;

	const unsigned char *read_ptr() const noexcept;

	unsigned char *write_ptr() noexcept;

	void flush();

	void close();
};

#endif // WIN32_BITMAP_H_
