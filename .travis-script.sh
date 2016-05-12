#!/bin/sh
set -e

echo "PWD: $(pwd)"
echo "CC: ${CC}"
echo "CXX: ${CXX}"
ls -l

# Do not execute build on coverity branch.
if [ "x$CC" = "xgcc-4.9" ]; then
  exit
fi

./autogen.sh
./configure --enable-simd --enable-testapp --enable-example --enable-unit-test
make clean
make
make test/unit_test
./test/unit_test
