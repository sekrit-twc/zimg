#include <climits>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include "argparse.h"

namespace {

const ArgparseOption HELP_OPTION_FULL = { OPTION_HELP, "?", "help", 0, nullptr, "print help message" };
const ArgparseOption HELP_OPTION_LONG_ONLY = { OPTION_HELP, nullptr, "help", 0, nullptr, "print help message" };

constexpr int HELP_INDENT = 32;


const char *get_short_name(const ArgparseOption &opt) noexcept
{
	if (opt.short_name)
		return opt.short_name;
	else if (opt.long_name)
		return opt.long_name;
	else
		return "";
}

const char *get_long_name(const ArgparseOption &opt) noexcept
{
	if (opt.long_name)
		return opt.long_name;
	else if (opt.short_name)
		return opt.short_name;
	else
		return "";
}

bool opt_has_param(OptionType type)
{
	switch (type) {
	case OPTION_INT:
	case OPTION_UINT:
	case OPTION_LONGLONG:
	case OPTION_ULONGLONG:
	case OPTION_FLOAT:
	case OPTION_STRING:
	case OPTION_USER1:
		return true;
	default:
		return false;
	}
}


struct Error {
	int code;
};

class OptionIterator {
public:
	typedef ptrdiff_t difference_type;
	typedef ArgparseOption value_type;
	typedef const ArgparseOption *pointer;
	typedef const ArgparseOption &reference;
	typedef std::bidirectional_iterator_tag iterator_category;
private:
	pointer m_opt;

	bool is_null() const noexcept { return !m_opt || m_opt->type == OPTION_NULL; }
public:
	explicit OptionIterator(pointer opt = nullptr) noexcept : m_opt{ opt } {}

	pointer get() const noexcept { return m_opt; }

	explicit operator bool() const noexcept { return !is_null(); }

	reference operator*() const noexcept { return *get(); }
	pointer operator->() const noexcept { return get(); }

	OptionIterator &operator++() noexcept { ++m_opt; return *this; }
	OptionIterator operator++(int) noexcept { OptionIterator ret = *this; ++*this; return ret; }

	OptionIterator &operator--() noexcept { --m_opt; return *this; }
	OptionIterator operator--(int) noexcept { OptionIterator ret = *this; ++*this; return ret; }

	bool operator==(const OptionIterator &other) const noexcept
	{
		return m_opt == other.m_opt || (is_null() && other.is_null());
	}

	bool operator!=(const OptionIterator &other) const noexcept { return !(*this == other); }
};

class OptionRange {
	const ArgparseOption *m_first;
public:
	explicit OptionRange(const ArgparseOption *first) noexcept : m_first{ first } {}

	OptionIterator begin() const noexcept { return OptionIterator{ m_first }; }
	OptionIterator end() const noexcept { return OptionIterator{}; }
};

class OptionMap {
	std::unordered_map<char, const ArgparseOption *> m_short;
	std::unordered_map<std::string, const ArgparseOption *> m_long;
public:
	void insert_opt(const ArgparseOption *opt)
	{
		if (opt->short_name)
			m_short[opt->short_name[0]] = opt;
		if (opt->long_name)
			m_long[opt->long_name] = opt;
	}

	const ArgparseOption *find_short(char c) const noexcept
	{
		auto it = m_short.find(c);
		return it == m_short.end() ? nullptr : it->second;
	}

	const ArgparseOption *find_long(const std::string &s) const noexcept
	{
		auto it = m_long.find(s);
		return it == m_long.end() ? nullptr : it->second;
	}
};


template <class T>
T stox(const char *s, size_t *pos);

template <>
int stox<int>(const char *s, size_t *pos) { return std::stoi(s, pos); }

template <>
unsigned stox<unsigned>(const char *s, size_t *pos)
{
#if ULONG_MAX > UINT_MAX
	unsigned long x = std::stoul(s, pos);
	return x > UINT_MAX ? throw std::out_of_range{ "integer out of range" } : static_cast<unsigned>(x);
#else
	return std::stoul(s, pos);
#endif
}

template <>
long long stox<long long>(const char *s, size_t *pos) { return std::stoll(s, pos); }

template <>
unsigned long long stox<unsigned long long>(const char *s, size_t *pos) { return std::stoull(s, pos); }


template <class T>
T parse_integer(const char *s)
{
	try {
		size_t pos;
		T x = stox<T>(s, &pos);

		if (s[pos] != '\0')
			throw std::invalid_argument{ "unparsed characters" };

		return static_cast<T>(x);
	} catch (const std::exception &e) {
		std::cerr << "error parsing integer from '" << s << "': " << e.what() << '\n';
		throw Error{ ARGPARSE_BAD_PARAMETER };
	}
}

double parse_double(const char *s)
{
	try {
		size_t pos;
		auto x = std::stod(s, &pos);

		if (s[pos] != '\0')
			throw std::invalid_argument{ "unparsed characters" };
		return x;
	} catch (const std::exception &e) {
		std::cerr << "error parsing float from: '" << s << "': " << e.what() << '\n';
		throw Error{ ARGPARSE_BAD_PARAMETER };
	}
}

void handle_switch(const ArgparseOption &opt, void *out, const char *param, bool negated)
{
	if (negated && opt.type != OPTION_FLAG && opt.type != OPTION_USER0) {
		std::cerr << "switch '" << get_long_name(opt) << "' can not be negated\n";
		throw Error{ ARGPARSE_BAD_PARAMETER };
	}
	if (opt_has_param(opt.type) && !param) {
		std::cerr << "switch '" << get_long_name(opt) << "' missing parameter\n";
		throw Error{ ARGPARSE_BAD_PARAMETER };
	}

	void *out_ptr = static_cast<char *>(out) + opt.offset;

	switch (opt.type) {
	case OPTION_FLAG:
		*static_cast<char *>(out_ptr) = !negated;
		break;
	case OPTION_HELP:
		throw Error{ ARGPARSE_HELP_MESSAGE };
	case OPTION_INCREMENT:
		*static_cast<int *>(out_ptr) += 1;
		break;
	case OPTION_DECREMENT:
		*static_cast<int *>(out_ptr) -= 1;
		break;
	case OPTION_INT:
		*static_cast<int *>(out_ptr) = parse_integer<int>(param);
		break;
	case OPTION_UINT:
		*static_cast<unsigned int *>(out_ptr) = parse_integer<unsigned>(param);
		break;
	case OPTION_LONGLONG:
		*static_cast<long long *>(out_ptr) = parse_integer<long long>(param);
		break;
	case OPTION_ULONGLONG:
		*static_cast<unsigned long long *>(out_ptr) = parse_integer<unsigned long long>(param);
		break;
	case OPTION_FLOAT:
		*static_cast<double *>(out_ptr) = parse_double(param);
		break;
	case OPTION_STRING:
		*static_cast<const char **>(out_ptr) = param;
		break;
	case OPTION_USER0:
	case OPTION_USER1:
		if (opt.func(&opt, out_ptr, param, negated))
			throw Error{ ARGPARSE_BAD_PARAMETER };
		break;
	case OPTION_NULL:
	default:
		throw std::logic_error{ "bad option type" };
	}
}

void handle_long_switch(const OptionMap &options, void *out, int argc, const char * const *argv, int *pos, const char *s, size_t len)
{
	const char *param = nullptr;

	if (const char *find = std::strchr(s, '=')) {
		param = find + 1;
		len = find - s;
	}

	const ArgparseOption *opt = nullptr;
	bool negated = false;

	std::string sw{ s, len };

	if ((opt = options.find_long(sw))) {
		// Ordinary switch: "--abc".
		negated = false;
	} else if (sw.find("no-") == 0 && (opt = options.find_long(sw.substr(3)))) {
		// Negated switch: "--no-abc".
		negated = true;
	} else {
		std::cerr << "unrecognized switch '--" << sw << "'\n";
		throw Error{ ARGPARSE_INVALID_SWITCH };
	}

	// Parameter as next argument: "--xyz 123".
	if (opt_has_param(opt->type) && !param && *pos < argc)
		param = argv[++*pos];

	handle_switch(*opt, out, param, negated);
};

void handle_short_switch(const OptionMap &options, void *out, int argc, const char * const *argv, int *pos, const char *s, size_t len)
{
	for (size_t i = 0; i < len; ++i) {
		const ArgparseOption *opt = options.find_short(s[i]);
		if (!opt) {
			std::cerr << "unrecognized switch '" << s[i] << "' (-" << s << ")\n";
			throw Error{ ARGPARSE_INVALID_SWITCH };
		}

		const char *param = nullptr;
		if (opt_has_param(opt->type)) {
			if (i < len - 1) {
				// Parameter in same argument: "-a3".
				param = s + i + 1;
			} else if (*pos < argc) {
				// Parameter as next argument: "-a 3".
				param = argv[++*pos];
			}

			// No more switches exist in the same argument.
			handle_switch(*opt, out, param, false);
			break;
		} else {
			handle_switch(*opt, out, nullptr, false);
		}
	}
};


void write(const char *s, size_t *len)
{
	std::cout << s;
	*len += std::strlen(s);
}

void print_switch(const ArgparseOption &opt)
{
	const char *short_name = opt.short_name;
	const char *long_name = opt.long_name;
	size_t len = 0;

	std::cout << '\t';

	if (long_name) {
		if (opt.type == OPTION_FLAG)
			write("--[no-]", &len);
		else
			write("--", &len);

		write(long_name, &len);
	}
	if (short_name) {
		if (long_name)
			write(" / ", &len);

		write("-", &len);
		write(short_name, &len);
	}

	if (opt.description) {
		for (; len < HELP_INDENT; ++len) {
			std::cout << ' ';
		}
		std::cout << opt.description;
	}
	std::cout << '\n';
}

void print_positional(const ArgparseOption &opt)
{
	const char *name = get_long_name(opt);
	size_t len = std::strlen(name);

	std::cout << '\t' << name;
	if (opt.description) {
		for (; len < HELP_INDENT; ++len) {
			std::cout << ' ';
		}
		std::cout << opt.description;
	}
	std::cout << '\n';
}

void print_help_message(const ArgparseCommandLine &cmd)
{
	bool has_help = false;
	bool has_q = false;

	// Print summary.
	if (cmd.summary)
		std::cout << cmd.program_name << ": " << cmd.summary << "\n\n";

	// Print short help.
	std::cout << "Usage: " << cmd.program_name << " [opts] ";
	for (const auto &opt : OptionRange{ cmd.positional }) {
		std::cout << get_short_name(opt) << ' ';
	}
	std::cout << '\n';

	// Print help for switches.
	std::cout << "Options:\n";
	for (const auto &opt : OptionRange{ cmd.switches }) {
		if (opt.type == OPTION_HELP || (opt.long_name && !std::strcmp(opt.long_name, "help")))
			has_help = true;
		if (opt.short_name && opt.short_name[1] == '?')
			has_q = true;

		print_switch(opt);
	}
	// Print built-in help option.
	if (!has_help) {
		if (!has_q)
			print_switch(HELP_OPTION_FULL);
		else
			print_switch(HELP_OPTION_LONG_ONLY);
	}

	// Print help for positional arguments.
	std::cout << "Arguments:\n";
	for (const auto &opt : OptionRange{ cmd.positional }) {
		print_positional(opt);
	}

	if (cmd.help_message)
		std::cout << '\n' << cmd.help_message << '\n';
}

} // namespace


int argparse_parse(const ArgparseCommandLine *cmd, void *out, int argc, char **argv)
{
	int ret = 0;

	try {
		OptionMap options;
		bool has_user_help = false;

		OptionIterator positional_cur{ cmd->positional };
		bool is_positional = false;
		int pos;

		for (const auto &opt : OptionRange{ cmd->switches }) {
			if (opt.type == OPTION_HELP || (opt.long_name && !std::strcmp(opt.long_name, "help")))
				has_user_help = true;

			options.insert_opt(&opt);
		}
		if (!has_user_help) {
			if (!options.find_short('?'))
				options.insert_opt(&HELP_OPTION_FULL);
			else
				options.insert_opt(&HELP_OPTION_LONG_ONLY);
		}

		for (pos = 1; pos < argc; ++pos) {
			const char *s = argv[pos];
			size_t len = std::strlen(s);

			if (!is_positional) {
				if (len > 2 && s[0] == '-' && s[1] == '-') {
					// Long form switch: "--[no-]xyz[=123]".
					handle_long_switch(options, out, argc, argv, &pos, s + 2, len - 2);
				} else if (len > 1 && s[0] == '-') {
					// Special end of switches marker: "--".
					if (s[1] == '-') {
						is_positional = true;
						continue;
					}
					// Short switch sequence: "-abc".
					handle_short_switch(options, out, argc, argv, &pos, s + 1, len - 1);
				} else {
					is_positional = true;
				}
			}
			if (is_positional) {
				// End of positional arguments or possible varargs.
				if (!positional_cur)
					break;

				// Next positional argument.
				handle_switch(*positional_cur++, out, s, false);
			}
		}
		if (positional_cur && positional_cur->type != OPTION_NULL) {
			// Insufficient positional arguments.
			std::cerr << "expected argument '" << get_long_name(*positional_cur) << "'\n";
			throw Error{ ARGPARSE_INSUFFICIENT_ARGS };
		}

		// Successful parse.
		ret = pos;
	} catch (const Error &e) {
		ret = e.code;
	} catch (const std::logic_error &e) {
		std::cerr << "malformed command line definition: " << e.what();
		std::terminate();
	} catch (const std::exception &e) {
		std::cerr << "error: " << e.what();
		ret = ARGPARSE_FATAL;
	}

	if (ret < 0 && ret != ARGPARSE_FATAL)
		print_help_message(*cmd);

	return ret;
}
