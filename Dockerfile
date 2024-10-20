FROM ubuntu:22.04

RUN apt update &&\
    apt upgrade -y &&\
    apt install clangd clang-format cmake git gnupg ninja-build libboost-program-options-dev libgtest-dev lsb-release python3-pip python-is-python3 software-properties-common wget -y &&\
    bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" &&\
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list.d/llvm.list &&\
    echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list.d/llvm.list &&\
    apt update &&\
    apt install clang-18 libmlir-18-dev llvm-18 mlir-18-tools -y &&\
    pip install -r requirements.txt
