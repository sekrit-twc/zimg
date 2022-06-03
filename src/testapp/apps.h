#pragma once

#ifndef APPS_H_
#define APPS_H_

#define PIXFMT_SPECIFIER_HELP_STR \
"Pixel format specifier: type[:fullrange[chroma][:depth]]\n" \
"fullrange: f=fullrange, l=limited\n" \
"chroma:    c=chroma, l=luma\n"

struct ArgparseOption;

#ifdef __cplusplus
extern "C" {
#endif

int arg_decode_cpu(const struct ArgparseOption *opt, void *out, const char *param, int negated);

int arg_decode_pixfmt(const struct ArgparseOption *opt, void *out, const char *param, int negated);

int colorspace_main(int argc, char **argv);
int cpuinfo_main(int argc, char **argv);
int depth_main(int argc, char **argv);
int graph_main(int argc, char **argv);
int graph2_main(int argc, char **argv);
int resize_main(int argc, char **argv);
int unresize_main(int argc, char **argv);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* APPS_H_ */
