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

namespace zimg {;

namespace _static_map {;

struct strcmp_less {
	bool operator()(const char *a, const char *b) const
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
	typedef const value_type *const_iterator;
private:
	struct value_compare {
		key_compare comp;

		bool operator()(const value_type &a, const value_type &b)
		{
			return comp(a.first, b.first);
		}
	};

	struct data_member : private Compare {
		std::array<value_type, Sz> array;

		data_member(const Compare &comp) : Compare(comp)
		{
		}

		key_compare get_key_comp() const
		{
			return *this;
		}

		value_compare get_value_comp() const
		{
			return{ *this };
		}
	};

	data_member m_head;
	size_t m_size;

	bool equiv(const Key &a, const Key &b) const
	{
		key_compare comp = m_head.get_key_comp();
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
	explicit static_map(std::initializer_list<value_type> init, const Compare &comp = Compare()) :
		m_head{ comp },
		m_size{ init.size() }
	{
		_zassert(init.size() <= Sz, "list size incorrect");

		std::copy(init.begin(), init.end(), m_head.array.begin());
		std::sort(m_head.array.begin(), m_head.array.begin() + init.size(), m_head.get_value_comp());
	}

	/**
	 * Get the number of elements in the map.
	 *
	 * @return number of elements
	 */
	size_type size() const
	{
		return m_size;
	}

	/**
	 * Get an iterator to the first map entry.
	 * Map entries are sorted.
	 *
	 * @return first entry
	 */
	const_iterator begin() const
	{
		return m_head.array.data();
	}

	/**
	 * Get an iterator to one past the last map entry.
	 *
	 * @return one past the end
	 */
	const_iterator end() const
	{
		return begin() + m_size;
	}

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

		if (it == end())
			throw std::out_of_range{ "key not found" };

		return it->second;
	}

	/**
	 * @see static_map::at(const Key &)
	 */
	const T &operator[](const Key &key) const
	{
		return at(key);
	}

	/**
	 * Search for a key in the map
	 *
	 * @param key searched key
	 * @return iterator to key if found, else end iterator
	 */
	const_iterator find(const Key &key) const
	{
		auto it = std::lower_bound(begin(), end(), value_type{ key, mapped_type{} }, m_head.get_value_comp());

		if (it == end() || !equiv(it->first, key))
			return end();
		else
			return it;
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
