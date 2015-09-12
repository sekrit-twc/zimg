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
public:
	MemoryMappedFile();

	MemoryMappedFile(const char *path, read_tag);

	MemoryMappedFile(const char *path, write_tag);

	MemoryMappedFile(const char *path, size_t size, create_tag);

	~MemoryMappedFile();

	size_t size() const;

	const void *read_ptr() const;

	void *write_ptr();

	void flush();

	void close();
};

#endif // MMAP_H_
