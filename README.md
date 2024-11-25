# FluidML

## Overview

`FluidML` is an open-source project for model inference acceleration on the CPU. It focuses on minimizing the peak memory overhead and reducing the inference time cost by the greedy algorithm and dynamic programming. It provides C++ and Python interfaces so users can integrate it into their projects efficiently.

## Getting Started

`FluidML` is built on `CMake`, and we recommend using `clang-18` and `clang++-18` as compilers to avoid unknown errors and bugs. It also depends on some third-party projects, including `LLVM` and `GTest`, so please install them in the compilation environment. We also provide a docker build script to simplify the environment management. Please run `docker build -t <IMAGE-NAME> .` to build the image, and then run `docker run -itd --name <CONTAINER-NAME> -v <LOCAL-WORKDIR>:<CONTAINER-WORKDIR> <IMAGE-NAME> bash` to start a container.

We provide several CMake options to control the product's attributes. The CMake configuration command is `cmake -G Ninja -DCMAKE_C_COMPILER=<C-COMPILER> -DCMAKE_CXX_COMPILER=<C++-COMPILER> -DCMAKE_BUILD_TYPE=<Debug|Release> -DBUILD_TESTS=<ON|OFF> -DBUILD_PYTHON=<ON|OFF> -DDP_DEBUG=<ON|OFF> -DUSE_LOGS=<ON|OFF> -S <SOURCE-DIR> -B <BUILD-DIR>`, and then run `cmake --build build` to start the compilation. We recommend to use `cmake -G Ninja -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DBUILD_PYTHON=ON -DDP_DEBUG=ON -DUSE_LOGS=ON -S <SOURCE-DIR> -B <BUILD-DIR>` to start it quickly as the first step.

## Benchmark and Test

We also provide benchmarks and tests for `FluidML`. It's controlled by `BUILD_DEBUG` option during the configuration. Then you can run all tests with `ctest`, and find executables in the `<BUILD-DIR>/benchmark`. Notice that the benchmarks may take several hours to run, so be patient and relax when running them. Drink a cup of coffee or watch an exciting movie!

We support the following models currently.

- [x] BERT
- [x] ConvBERT
- [x] EfficientNet
- [x] GPT-NEOX
- [x] I-BRET
- [x] VGG

You can find them in the benchmarks.

## Citation

Please refer to the paper [FluidML: Fast and Memory Efficient Inference Optimization](https://arxiv.org/abs/2411.09242) on Arxiv.
