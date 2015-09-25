#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef enum OptionType {
	OPTION_BOOL,
	OPTION_TRUE,
	OPTION_FALSE,
	OPTION_INTEGER,
	OPTION_UINTEGER,
	OPTION_FLOAT,
	OPTION_STRING,
	OPTION_USER
} OptionType;

typedef struct ArgparseOption {
	OptionType type;
	const char *short_name;
	const char *long_name;
	size_t offset;
	int (*func)(const struct ArgparseOption *opt, void *out, int argc, char **argv);
	const char *description;
} ArgparseOption;

typedef struct ArgparseCommandLine {
	const ArgparseOption *switches;
	size_t num_switches;
	const ArgparseOption *positional;
	size_t num_positional;
	const char *program_name;
	const char *summary;
	const char *help_message;
} ArgparseCommandLine;

#define ARGPARSE_HELP  1
#define ARGPARSE_ERROR 2
#define ARGPARSE_FATAL 3

int argparse_parse(const ArgparseCommandLine *cmd, void *out, int argc, char **argv);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
