#pragma once

#ifndef ZIMG_STATIC_MAP_H_
#define ZIMG_STATIC_MAP_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include "zassert.h"

namespace zimg {

namespace _static_map {

struct strcmp_less {
	bool operator()(const char *a, const char *b) const noexcept
	{
		return strcmp(a, b) < 0;
	}
};

} // namespace _static_map


/**
 * Fixed size read-only map.
 *
 * @tparam Key key type
 * @tparam T mapped type
 * @tparam Sz maximum number of elements
 * @tparam Compare comparator
 */
template <class Key, class T, size_t Sz, class Compare = std::less<Key>>
class static_map {
public:
	typedef Key key_type;
	typedef T mapped_type;
	typedef std::pair<Key, T> value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef Compare key_compare;
	typedef const value_type &reference;
	typedef reference const_reference;
	typedef const value_type *pointer;
	typedef pointer const_pointer;
	typedef const_pointer iterator;
	typedef iterator const_iterator;
private:
	struct value_compare {
		const key_compare &comp;

		bool operator()(const value_type &a, const value_type &b) noexcept { return comp(a.first, b.first); }
	};

	struct data_member : private Compare {
		std::array<value_type, Sz> array;

		data_member(const Compare &comp) noexcept : Compare(comp) {}

		const key_compare &get_key_comp() const noexcept { return *this; }
		value_compare get_value_comp() const noexcept { return{ *this }; }
	};

	data_member m_head;
	size_t m_size;

	bool equiv(const Key &a, const Key &b) const noexcept
	{
		const key_compare &comp = m_head.get_key_comp();
		return !comp(a, b) && !comp(b, a);
	}
public:
	/**
	 * Construct a static_map from brace-enclosed initializers.
	 *
	 * The initializer list must be shorter than the map size and may not
	 * contain duplicate keys.
	 *
	 * @param init initializer list
	 * @param comp comparator
	 */
	explicit static_map(std::initializer_list<value_type> init, const Compare &comp = Compare()) noexcept :
		m_head{ comp },
		m_size{ init.size() }
	{
		zassert(init.size() <= Sz, "list size incorrect");

		std::copy(init.begin(), init.end(), m_head.array.begin());
		std::sort(m_head.array.begin(), m_head.array.begin() + init.size(), m_head.get_value_comp());
	}

	/**
	 * Get the number of elements in the map.
	 *
	 * @return number of elements
	 */
	size_type size() const noexcept { return m_size; }

	/**
	 * Get the number of elements in the largest possible map.
	 */
	size_type max_size() const noexcept { return SIZE_MAX; }

	/**
	 * Get an iterator to the first map entry.
	 * Map entries are sorted.
	 *
	 * @return first entry
	 */
	const_iterator begin() const noexcept { return m_head.array.data(); }

	/**
	 * Get an iterator to one past the last map entry.
	 *
	 * @return one past the end
	 */
	const_iterator end() const noexcept { return begin() + m_size; }

	/**
	 * @see begin
	 */
	const_iterator cbegin() const noexcept { return begin(); }

	/**
	 * @see end
	 */
	const_iterator cend() const noexcept { return end(); }

	/**
	 * Get the mapped value corresponding to a key.
	 *
	 * @param key searched key
	 * @param value mapped value
	 * @throw std::out_of_range if key not present
	 */
	const T &at(const Key &key) const
	{
		const auto it = find(key);
		return *(it == end() ? throw std::out_of_range{ "key not found" } : &it->second);
	}

	/**
	 * @see at
	 */
	const T &operator[](const Key &key) const { return at(key); }

	/**
	 * Search for a key in the map
	 *
	 * @param key searched key
	 * @return iterator to key if found, else end iterator
	 */
	const_iterator find(const Key &key) const noexcept
	{
		const_iterator it = std::lower_bound(begin(), end(), value_type{ key, mapped_type{} }, m_head.get_value_comp());
		return (it == end() || !equiv(it->first, key)) ? end() : it;
	}
};

/**
 * Map keyed on null-terminated strings.
 *
 * @tparam T mapped type
 * @tparam Sz maximum number of elements
 */
template <class T, size_t Sz>
using static_string_map = static_map<const char *, T, Sz, _static_map::strcmp_less>;

} // namespace zimg

#endif // ZIMG_STATIC_MAP_H_
