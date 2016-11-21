#pragma once

#ifndef MMAP_H_
#define MMAP_H_

#include <cstddef>
#include <memory>

class MemoryMappedFile {
	class impl;

	struct read_tag {};
	struct write_tag {};
	struct create_tag {};
public:
	static const read_tag READ_TAG;
	static const write_tag WRITE_TAG;
	static const create_tag CREATE_TAG;
private:
	std::unique_ptr<impl> m_impl;

	impl *get_impl() { return m_impl.get(); }
	const impl *get_impl() const { return m_impl.get(); }
public:
	MemoryMappedFile() noexcept;

	MemoryMappedFile(MemoryMappedFile &&other) noexcept;

	MemoryMappedFile(const char *path, read_tag);

	MemoryMappedFile(const char *path, write_tag);

	MemoryMappedFile(const char *path, size_t size, create_tag);

	~MemoryMappedFile();

	MemoryMappedFile &operator=(MemoryMappedFile &&other) noexcept;

	size_t size() const noexcept;

	const void *read_ptr() const noexcept;

	void *write_ptr() noexcept;

	void flush();

	void close();
};

#endif // MMAP_H_
