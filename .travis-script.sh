#!/bin/sh
set -e

NPROC=$(nproc --all)

echo "PWD: $(pwd)"
echo "CC: ${CC}"
echo "CXX: ${CXX}"
echo "TRAVIS_BRANCH: ${TRAVIS_BRANCH}"
echo "NPROC: ${NPROC}"
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
make "-j${NPROC}"
make "-j${NPROC}" test/unit_test
./testapp cpuinfo
./test/unit_test
