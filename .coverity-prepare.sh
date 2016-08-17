cov-configure --comptype gcc --compiler ${CC} --template
./autogen.sh
./configure --enable-simd --enable-testapp --enable-example
make clean
