#!/bin/sh
set -e

echo "PWD: $(pwd)"
echo "CC: ${CC}"
echo "CXX: ${CXX}"
echo "TRAVIS_BRANCH: ${TRAVIS_BRANCH}"
ls -l

# Do not execute build on coverity branch.
if [ "x$COVERITY_SCAN_BRANCH" = "x1" -o "x$TRAVIS_BRANCH" = "xcoverity_scan" ]; then
  test -f cov-int/build-log.txt && tail -n 100 cov-int/build-log.txt || true
  test -f cov-int/scm_log.txt && tail -n 100 cov-int/scm_log.txt || true
  exit
fi

./autogen.sh
./configure --enable-simd --enable-testapp --enable-example --enable-unit-test
make clean
make -j2
make -j2 test/unit_test
./test/unit_test
