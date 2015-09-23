#pragma once

#ifndef APPS_H_
#define APPS_H_

struct ArgparseOption;

int arg_decode_cpu(const struct ArgparseOption *opt, void *out, int argc, char **argv);

#if 0
int colorspace_main(int argc, const char **argv);

int depth_main(int argc, const char **argv);

int resize_main(int argc, const char **argv);

int unresize_main(int argc, const char **argv);
#endif

#endif /* APPS_H_ */
