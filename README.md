# FluidML

Please use the Dockerfile in the root, and run `cmake -G Ninja -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_PYTHON=OFF -DDP_DEBUG=OFF -DUSE_LOGS=OFF ..` to build the project. You can run the benchmarks in `build/benchmark/` to test it.
