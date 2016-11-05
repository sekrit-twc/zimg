#ifndef ARGPARSE_H_
#define ARGPARSE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OptionType {
	OPTION_NULL,
	OPTION_FLAG,
	OPTION_HELP,
	OPTION_INCREMENT,
	OPTION_DECREMENT,
	OPTION_INT,
	OPTION_UINT,
	OPTION_LONGLONG,
	OPTION_ULONGLONG,
	OPTION_FLOAT,
	OPTION_STRING,
	OPTION_USER0,
	OPTION_USER1,
} OptionType;

typedef struct ArgparseOption {
	OptionType type;
	const char *short_name;
	const char *long_name;
	size_t offset;
	int (*func)(const struct ArgparseOption *opt, void *out, const char *param, int negated);
	const char *description;
} ArgparseOption;

typedef struct ArgparseCommandLine {
	const ArgparseOption *switches; /* Terminated by OPTION_NULL. */
	const ArgparseOption *positional; /* Terminated by OPTION_NULL. */
	const char *program_name;
	const char *summary;
	const char *help_message;
} ArgparseCommandLine;

enum {
	ARGPARSE_HELP_MESSAGE = -1,
	ARGPARSE_INSUFFICIENT_ARGS = -2,
	ARGPARSE_INVALID_SWITCH = -3,
	ARGPARSE_BAD_PARAMETER = -4,
	ARGPARSE_FATAL = -128
};

/* Returns number of arguments parsed, or negative error code. */
int argparse_parse(const ArgparseCommandLine *cmd, void *out, int argc, char **argv);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ARGPARSE_H_ */
