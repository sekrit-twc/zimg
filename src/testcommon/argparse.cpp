#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <stddef.h>
#include "argparse.h"

namespace {;

int handle_argument_bool(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	std::string str = *argv;
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	int *out_p = static_cast<int *>(out);
	int value = 0;

	if (str == "true" || str == "1")
		value = 1;
	else if (str == "false" || str =="0")
		value = 0;
	else
		return -1;

	*out_p = value;
	return 1;
}

int handle_argument_true(const ArgparseOption *, void *out, int, char **)
{
	int *out_p = static_cast<int *>(out);
	*out_p = 1;
	return 0;
}

int handle_argument_false(const ArgparseOption *, void *out, int, char **)
{
	int *out_p = static_cast<int *>(out);
	*out_p = 0;
	return 0;
}

int handle_argument_integer(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	try {
		int *out_p = static_cast<int *>(out);
		*out_p = std::stoi(*argv);
		return 1;
	} catch (const std::logic_error &) {
		return -1;
	}
}

int handle_argument_uinteger(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	try {
		unsigned *out_p = static_cast<unsigned *>(out);
		unsigned long x = std::stoul(*argv);

		if (x > UINT_MAX)
			throw std::out_of_range{ "" };

		*out_p = static_cast<unsigned>(x);
		return 1;
	} catch (const std::logic_error &) {
		return -1;
	}
}

int handle_argument_float(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	try {
		double *out_p = static_cast<double *>(out);
		*out_p = std::stod(*argv);
		return 1;
	} catch (const std::logic_error &) {
		return -1;
	}
}

int handle_argument_string(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	const char **out_p = static_cast<const char **>(out);
	*out_p = *argv;
	return 1;
}

int handle_argument(const ArgparseOption *opt, void *out, int argc, char **argv)
{
	void *out_p = (char *)out + opt->offset;

	switch (opt->type) {
	case OPTION_BOOL:
		return handle_argument_bool(opt, out_p, argc, argv);
	case OPTION_TRUE:
		return handle_argument_true(opt, out_p, argc, argv);
	case OPTION_FALSE:
		return handle_argument_false(opt, out_p, argc, argv);
	case OPTION_INTEGER:
		return handle_argument_integer(opt, out_p, argc, argv);
	case OPTION_UINTEGER:
		return handle_argument_uinteger(opt, out_p, argc, argv);
	case OPTION_FLOAT:
		return handle_argument_float(opt, out_p, argc, argv);
	case OPTION_STRING:
		return handle_argument_string(opt, out_p, argc, argv);
	case OPTION_USER:
		return opt->func(opt, out_p, argc, argv);
	default:
		throw std::invalid_argument{ "bad argument type" };
	}
}

void print_usage(const ArgparseCommandLine *cmd)
{
	if (cmd->summary)
		std::cout << cmd->program_name << ": " << cmd->summary << "\n\n";

	std::cout << "Usage: " << cmd->program_name << " [opts] ";
	for (size_t i = 0; i < cmd->num_positional; ++i) {
		const ArgparseOption *opt = cmd->positional + i;
		const char *name = opt->short_name ? opt->short_name : opt->long_name;
		std::cout << name << ' ';
	}
	std::cout << "\n";

	std::cout << "Options:\n";
	for (size_t i = 0; i < cmd->num_switches; ++i) {
		const ArgparseOption *opt = cmd->switches + i;
		const char *short_name = opt->short_name;
		const char *long_name = opt->long_name;
		size_t len;

		std::cout << "\t";

		if (short_name && long_name) {
			std::cout << "--" << long_name << " / -" << short_name;
			len = strlen(short_name) + strlen(long_name) + 2 + 4;
		} else if (short_name) {
			std::cout << "-" << short_name;
			len = strlen(short_name) + 1;
		} else {
			std::cout << "--" << long_name;
			len = strlen(long_name) + 2;
		}

		if (opt->description) {
			for (; len < 32; ++len) {
				std::cout << ' ';
			}
			std::cout << opt->description;
		}
		std::cout << '\n';
	}
	std::cout << "Arguments:\n";
	for (size_t i = 0; i < cmd->num_positional; ++i) {
		const ArgparseOption *opt = cmd->positional + i;
		const char *name = opt->long_name;
		size_t len = strlen(name);

		std::cout << "\t" << name;
		if (opt->description) {
			for (; len < 32; ++len) {
				std::cout << ' ';
			}
			std::cout << opt->description;
		}
		std::cout << '\n';
	}

	if (cmd->help_message)
		std::cout << '\n' << cmd->help_message << '\n';
}

template <class MapType, class K = typename MapType::key_type, class T = typename MapType::mapped_type>
T map_find_default(const MapType &map, const K &key, const T &default_value = T{})
{
	auto it = map.find(key);
	return it == map.end() ? default_value : it->second;
}

} // namespace


int argparse_parse(const ArgparseCommandLine *cmd, void *out, int argc, char **argv)
{
	std::unordered_map<std::string, const ArgparseOption *> switch_short_map;
	std::unordered_map<std::string, const ArgparseOption *> switch_long_map;
	bool has_short_help = false;
	int error = 0;

	for (size_t i = 0; i < cmd->num_switches; ++i) {
		const ArgparseOption *opt = cmd->switches + i;

		if (opt->short_name)
			switch_short_map[opt->short_name] = opt;
		if (opt->long_name && strcmp(opt->long_name, "help"))
			switch_long_map[opt->long_name] = opt;
	}
	has_short_help = (switch_short_map.find("h") == switch_short_map.end());

	try {
		int arg_index = 1;
		size_t positional_count = 0;
		bool positional_flag = false;
		std::string str;

		while (arg_index < argc) {
			const ArgparseOption *opt = nullptr;
			int adjust;
			int ret;

			str = argv[arg_index];

			if (!positional_flag) {
				if ((has_short_help && str == "-h") || str == "--help") {
					error = ARGPARSE_HELP;
					break;
				}
				if (!opt && str.find("--no-") == 0)
					opt = map_find_default(switch_long_map, str.substr(5));
				if (!opt && str.find("--") == 0)
					opt = map_find_default(switch_long_map, str.substr(2));
				if (!opt && str.find("-") == 0)
					opt = map_find_default(switch_short_map, str.substr(1));

				if (!opt)
					positional_flag = true;
			}

			if (positional_flag) {
				if (positional_count >= cmd->num_positional) {
					std::cerr << "too many positional arguments\n";
					error = ARGPARSE_ERROR;
					break;
				}

				opt = cmd->positional + positional_count++;
			}

			adjust = positional_flag ? 0 : 1;

			if ((ret = handle_argument(opt, out, argc - arg_index - adjust, argv + arg_index + adjust)) < 0) {
				std::cerr << "error parsing argument: " << (opt->short_name ? opt->short_name : opt->long_name) << '\n';
				error = ARGPARSE_ERROR;
				break;
			}
			if (positional_flag && ret == 0)
				throw std::logic_error{ "positional argument must advance argument list" };

			arg_index += ret + adjust;
		}

		if (arg_index >= argc && positional_count != cmd->num_positional) {
			const ArgparseOption *opt = cmd->positional + positional_count;

			std::cerr << "missing positional argument: ";
			std::cerr << (opt->short_name ? opt->short_name : opt->long_name) << '\n';
			error = ARGPARSE_ERROR;
		}
	} catch (const std::logic_error &e) {
		std::cerr << e.what() << '\n';
		error = ARGPARSE_FATAL;
	}

	if (error == ARGPARSE_HELP || error == ARGPARSE_ERROR)
		print_usage(cmd);

	return error;
}
