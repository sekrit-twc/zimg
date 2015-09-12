#include <climits>
#include <cstring>
#include <system_error>

#ifdef _WIN32
  #include <Windows.h>
#endif // _WIN32

#include "mmap.h"

namespace {;

#ifdef _WIN32
namespace win32 {;

class close_handle {
	struct handle {
		::HANDLE h = INVALID_HANDLE_VALUE;

		handle(nullptr_t x = nullptr)
		{
		}

		handle(::HANDLE h) : h{ h }
		{
		}

		operator ::HANDLE() const
		{
			return h;
		}

		bool operator==(nullptr_t) const
		{
			return h != 0 && h != INVALID_HANDLE_VALUE;
		}

		bool operator!=(nullptr_t) const
		{
			return !(*this == nullptr);
		}
	};

public:
	typedef handle pointer;

	void operator()(handle h)
	{
		if (h)
			::CloseHandle(h);
	}
};

typedef std::unique_ptr<void, win32::close_handle> handle_uptr;

struct unmap_view_of_file {
	void operator()(void *ptr)
	{
		::UnmapViewOfFile(ptr);
	}
};

void trap_error(const char *msg = "")
{
	std::error_code code{ (int)::GetLastError(), std::system_category() };
	throw std::system_error{ code, msg };
}

void utf8_to_wchar(wchar_t unicode_path[MAX_PATH], const char *path)
{
	size_t path_len = strlen(path);

	if (path_len > MAX_PATH)
		throw std::logic_error{ "path too long" };

	if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, (int)path_len, unicode_path, MAX_PATH * sizeof(wchar_t)) == 0)
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
	if (::CloseHandle(file_handle) == 0)
		win32::trap_error("error closing file handle");

	file_handle_uptr.release();
}

} // namespace win32
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
		m_size = (size_t)file_size.QuadPart;
	}
public:
	size_t size() const
	{
		return m_size;
	}

	const void *read_ptr() const
	{
		return m_map_view.get();
	}

	void *write_ptr() const
	{
		return m_writable ? m_map_view.get() : nullptr;
	}

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

	void unmap()
	{
		m_map_view.release();
	}
};
#endif


const MemoryMappedFile::read_tag MemoryMappedFile::READ_TAG{};
const MemoryMappedFile::write_tag MemoryMappedFile::WRITE_TAG{};
const MemoryMappedFile::create_tag MemoryMappedFile::CREATE_TAG{};

MemoryMappedFile::MemoryMappedFile()
{
}

MemoryMappedFile::MemoryMappedFile(const char *path, read_tag) : m_impl{ new impl{} }
{
	m_impl->map_read(path);
}

MemoryMappedFile::MemoryMappedFile(const char *path, write_tag) : m_impl{ new impl{} }
{
	m_impl->map_write(path);
}

MemoryMappedFile::MemoryMappedFile(const char *path, size_t size, create_tag) : m_impl{ new impl{} }
{
	m_impl->map_create(path, size);
}

MemoryMappedFile::~MemoryMappedFile()
{
}

size_t MemoryMappedFile::size() const
{
	return m_impl->size();
}

const void *MemoryMappedFile::read_ptr() const
{
	return m_impl->read_ptr();
}

void *MemoryMappedFile::write_ptr()
{
	return m_impl->write_ptr();
}

void MemoryMappedFile::flush()
{
	m_impl->flush();
}

void MemoryMappedFile::close()
{
	m_impl->flush();
	m_impl->unmap();
}
