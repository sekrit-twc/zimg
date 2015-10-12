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
public:
	WindowsBitmap(const char *path, read_tag);

	WindowsBitmap(const char *path, write_tag);

	WindowsBitmap(const char *path, int width, int height, int bit_count);

	~WindowsBitmap();

	ptrdiff_t stride() const;

	int width() const;

	int height() const;

	int bit_count() const;

	const unsigned char *read_ptr() const;

	unsigned char *write_ptr();

	void flush();

	void close();
};

#endif // WIN32_BITMAP_H_
