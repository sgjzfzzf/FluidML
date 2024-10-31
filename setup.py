import os
import setuptools
import setuptools.command.build_ext
import shutil
import subprocess
from typing import List, Tuple


class CMakeExtension(setuptools.Extension):
    def __init__(self, name):
        setuptools.Extension.__init__(self, name, [])


class CMakeBuild(setuptools.command.build_ext.build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        source_dir: str = os.path.abspath(os.path.dirname(__file__))
        build_dir: str = os.path.join(source_dir, "build")
        ext_path: str = os.path.abspath(self.get_ext_fullpath(ext.name))
        ext_dir: str = os.path.dirname(ext_path)
        os.makedirs(f"{build_dir}", exist_ok=True)
        subprocess.check_call(["cmake", "--version"])
        cc: Tuple[str | None] = os.environ.get("CC")
        cxx: Tuple[str | None] = os.environ.get("CXX")
        cmake_args: List[str] = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PYTHON=ON",
            "-DDP_DEBUG=OFF",
            "-DUSE_LOGS=OFF",
        ]
        if cc:
            cmake_args += [f"-DCMAKE_C_COMPILER={cc}"]
        if cxx:
            cmake_args += [f"-DCMAKE_CXX_COMPILER={cxx}"]
        try:
            subprocess.check_call(["ninja", "--version"])
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        else:
            cmake_args += ["-G", "Ninja"]
        subprocess.check_call(
            [
                "cmake",
                "-S",
                f"{source_dir}",
                "-B",
                f"{build_dir}",
            ]
            + cmake_args,
            cwd=build_dir,
        )
        subprocess.check_call(
            [
                "cmake",
                "--build",
                f"{build_dir}",
                "--",
                f"-j{os.cpu_count()}",
            ],
            cwd=build_dir,
        )
        pyi_filename = f"{ext.name}.pyi"
        shutil.copyfile(
            os.path.join(source_dir, "src", "python", pyi_filename),
            os.path.join(ext_dir, pyi_filename),
        )


if __name__ == "__main__":
    name: str = "fluidml"
    setuptools.setup(
        name=name,
        packages=setuptools.find_packages(),
        ext_modules=[CMakeExtension(name)],
        cmdclass={"build_ext": CMakeBuild},
    )
