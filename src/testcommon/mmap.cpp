#ifndef _WIN32
  #define _FILE_OFFSET_BITS 64
#endif // _WIN32

#include <climits>
#include <cstring>
#include <system_error>

#ifdef _WIN32
  #include <Windows.h>
#else
  #include <cerrno>
  #include <cstdint>

  #include <fcntl.h>
  #include <unistd.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <sys/types.h>
#endif // _WIN32

#include "mmap.h"

using std::nullptr_t;

namespace {

#ifdef _WIN32
namespace win32 {

class close_handle {
	struct handle {
		::HANDLE h = INVALID_HANDLE_VALUE;

		handle(nullptr_t x = nullptr) noexcept {}
		handle(::HANDLE h) noexcept : h{ h } {}

		operator ::HANDLE() const noexcept { return h; }

		bool operator==(nullptr_t) const noexcept { return h != 0 && h != INVALID_HANDLE_VALUE; }
		bool operator!=(nullptr_t) const noexcept { return !(*this == nullptr); }
	};
public:
	typedef handle pointer;

	void operator()(handle h) noexcept { ::CloseHandle(h); }
};

typedef std::unique_ptr<void, win32::close_handle> handle_uptr;

struct unmap_view_of_file {
	void operator()(void *ptr) noexcept { ::UnmapViewOfFile(ptr); }
};

void trap_error(const char *msg = "")
{
	std::error_code code{ (int)::GetLastError(), std::system_category() };
	throw std::system_error{ code, msg };
}

void utf8_to_wchar(wchar_t unicode_path[MAX_PATH], const char *path)
{
	if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, unicode_path, MAX_PATH * sizeof(wchar_t)) == 0)
		win32::trap_error("error converting path to UTF-16");
}

void create_new_file(const char *path, size_t size)
{
	wchar_t unicode_path[MAX_PATH] = { 0 };
	handle_uptr file_handle_uptr;
	::HANDLE file_handle;
	::LARGE_INTEGER file_ptr;

	win32::utf8_to_wchar(unicode_path, path);

	if ((file_handle = ::CreateFileW(unicode_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr)) == INVALID_HANDLE_VALUE)
		win32::trap_error("error opening file");

	file_handle_uptr.reset(file_handle);
	file_ptr.QuadPart = size;

	if (::SetFilePointerEx(file_handle, file_ptr, nullptr, FILE_BEGIN) == 0)
		win32::trap_error("error setting file pointer");
	if (::SetEndOfFile(file_handle) == 0)
		win32::trap_error("error setting end of file");
	if (::CloseHandle(file_handle_uptr.release()) == 0)
		win32::trap_error("error closing file handle");
}

} // namespace win32
#else
namespace posix {

class close_fd {
	struct descriptor {
		int fd = -1;

		descriptor(nullptr_t x = nullptr) noexcept {}
		descriptor(int fd) noexcept : fd{ fd } {}

		operator int() const noexcept { return fd; }

		bool operator==(nullptr_t) const noexcept { return fd < 0; }
		bool operator!=(nullptr_t) const noexcept { return !(*this == nullptr); }
	};
public:
	typedef descriptor pointer;

	void operator()(descriptor x) noexcept { ::close(x); }
};

class munmap_file {
	struct map_pointer {
		void *ptr = MAP_FAILED;

		map_pointer(nullptr_t x = nullptr) noexcept {}
		map_pointer(void *ptr) noexcept : ptr{ ptr } {}

		operator void *() const noexcept { return ptr; }

		bool operator==(nullptr_t) const noexcept { return ptr == MAP_FAILED; }
		bool operator!=(nullptr_t) const noexcept { return !(*this == nullptr); }
	};
public:
	typedef map_pointer pointer;

	size_t size = 0;

	void operator()(map_pointer ptr) noexcept { ::munmap(ptr, size); }
};

void trap_error(const char *msg = "")
{
	std::error_code code{ errno, std::system_category() };
	throw std::system_error{ code, msg };
}

off_t get_file_size(int fd)
{
	off_t pos = -1;
	off_t ret;

	if ((pos = ::lseek(fd, 0, SEEK_CUR)) < 0)
		posix::trap_error("error getting file position");
	if ((ret = ::lseek(fd, 0, SEEK_END)) < 0)
		posix::trap_error("error seeking in file");
	if ((pos = ::lseek(fd, pos, SEEK_SET)) < 0)
		posix::trap_error("error setting file position");

	return ret;
}

void create_new_file(const char *path, size_t size)
{
	std::unique_ptr<void, close_fd> fd_uptr;
	int fd;

	if ((fd = ::creat(path, 00666)) < 0)
		posix::trap_error("error creating file");

	fd_uptr.reset(fd);

	if (ftruncate(fd, size) < 0)
		posix::trap_error("error truncating file");
	if (close(fd) < 0)
		posix::trap_error("error closing file");
}

} // namespace posix
#endif // _WIN32

} // namespace


#ifdef _WIN32
class MemoryMappedFile::impl {
	std::unique_ptr<void, win32::unmap_view_of_file> m_map_view;

	size_t m_size;
	bool m_writable;

	void map_file(const char *path, DWORD desired_access1, DWORD share_mode, DWORD protect, DWORD desired_access2)
	{
		win32::handle_uptr file_handle_uptr;
		win32::handle_uptr mapping_handle_uptr;

		wchar_t unicode_path[MAX_PATH] = { 0 };
		::HANDLE file_handle;
		::HANDLE mapping_handle;
		void *map_view;
		::LARGE_INTEGER file_size;

		win32::utf8_to_wchar(unicode_path, path);

		if ((file_handle = ::CreateFileW(unicode_path, desired_access1, share_mode, nullptr, OPEN_EXISTING, 0, nullptr)) == INVALID_HANDLE_VALUE)
			win32::trap_error("error opening file");

		file_handle_uptr.reset(file_handle);

		if (GetFileSizeEx(file_handle, &file_size) == 0)
			win32::trap_error("error getting file size");
		if (file_size.QuadPart > SIZE_MAX)
			throw std::runtime_error{ "file too large to map" };

		if ((mapping_handle = ::CreateFileMappingW(file_handle, nullptr, protect, file_size.HighPart, file_size.LowPart, nullptr)) == NULL)
			win32::trap_error("error creating file mapping");

		mapping_handle_uptr.reset(mapping_handle);

		if (!(map_view = ::MapViewOfFile(mapping_handle, desired_access2, 0, 0, 0)))
			win32::trap_error("error mapping view of file");

		m_map_view.reset(map_view);
		m_size = static_cast<size_t>(file_size.QuadPart);
	}
public:
	impl() noexcept : m_size{}, m_writable{} {}

	size_t size() const noexcept { return m_size; }

	const void *read_ptr() const noexcept { return m_map_view.get(); }

	void *write_ptr() const noexcept { return m_writable ? m_map_view.get() : nullptr; }

	void map_read(const char *path)
	{
		map_file(path, GENERIC_READ, FILE_SHARE_READ, PAGE_READONLY, FILE_MAP_READ);
		m_writable = false;
	}

	void map_write(const char *path)
	{
		map_file(path, GENERIC_READ | GENERIC_WRITE, 0, PAGE_READWRITE, FILE_MAP_WRITE);
		m_writable = true;
	}

	void map_create(const char *path, size_t size)
	{
		win32::create_new_file(path, size);
		map_write(path);
	}

	void flush()
	{
		if (m_map_view && ::FlushViewOfFile(m_map_view.get(), 0) == 0)
			win32::trap_error("error flushing file");
	}

	void close()
	{
		flush();
		m_map_view.reset();
	}
};
#else
class MemoryMappedFile::impl {
	std::unique_ptr<void, posix::munmap_file> m_ptr;

	size_t m_size;
	bool m_writable;

	void map_file(const char *path, int open_flags, int prot, int mmap_flags)
	{
		std::unique_ptr<void, posix::close_fd> fd_uptr;
		int fd;
		off_t file_size;
		void *ptr;

		if ((fd = ::open(path, open_flags)) < 0)
			posix::trap_error("error opening file");

		fd_uptr.reset(fd);

		if ((file_size = posix::get_file_size(fd)) > PTRDIFF_MAX)
			throw std::runtime_error{ "file too large to map" };

		if ((ptr = ::mmap(nullptr, file_size, prot, mmap_flags, fd, 0)) == MAP_FAILED)
			posix::trap_error("error mapping file");

		m_ptr.reset(ptr);
		m_ptr.get_deleter().size = static_cast<size_t>(file_size);
	}
public:
	impl() noexcept : m_size{}, m_writable{} {}

	size_t size() const noexcept { return m_ptr.get_deleter().size; }

	const void *read_ptr() const noexcept { return m_ptr.get(); }

	void *write_ptr() const noexcept { return m_writable ? m_ptr.get() : nullptr; }

	void map_read(const char *path)
	{
		map_file(path, O_RDONLY, PROT_READ, MAP_PRIVATE);
		m_writable = false;
	}

	void map_write(const char *path)
	{
		map_file(path, O_RDWR, PROT_READ | PROT_WRITE, MAP_SHARED);
		m_writable = true;
	}

	void map_create(const char *path, size_t size)
	{
		posix::create_new_file(path, size);
		map_write(path);
	}

	void flush()
	{
		if (::msync(m_ptr.get(), m_size, MS_SYNC))
			posix::trap_error("error flushing file");
	}

	void close()
	{
		flush();
		m_ptr.reset();
	}
};
#endif // _WIN32


const MemoryMappedFile::read_tag MemoryMappedFile::READ_TAG{};
const MemoryMappedFile::write_tag MemoryMappedFile::WRITE_TAG{};
const MemoryMappedFile::create_tag MemoryMappedFile::CREATE_TAG{};

MemoryMappedFile::MemoryMappedFile() noexcept = default;

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile &&other) noexcept = default;

MemoryMappedFile::MemoryMappedFile(const char *path, read_tag) : m_impl{ new impl{} }
{
	get_impl()->map_read(path);
}

MemoryMappedFile::MemoryMappedFile(const char *path, write_tag) : m_impl{ new impl{} }
{
	get_impl()->map_write(path);
}

MemoryMappedFile::MemoryMappedFile(const char *path, size_t size, create_tag) : m_impl{ new impl{} }
{
	get_impl()->map_create(path, size);
}

MemoryMappedFile::~MemoryMappedFile() = default;

MemoryMappedFile &MemoryMappedFile::operator=(MemoryMappedFile &&other) noexcept = default;

size_t MemoryMappedFile::size() const noexcept { return get_impl()->size(); }

const void *MemoryMappedFile::read_ptr() const noexcept { return get_impl()->read_ptr(); }

void *MemoryMappedFile::write_ptr() noexcept { return get_impl()->write_ptr(); }

void MemoryMappedFile::flush() { get_impl()->flush(); }

void MemoryMappedFile::close() { get_impl()->close(); }
