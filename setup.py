"""构建 llm_infer._C。

    pip install -e . --no-build-isolation

csrc/cpu 下的算子总是编译；csrc/cuda 下的 .cu 文件在检测到 CUDA 时一起编译
（设置 LLM_INFER_NO_CUDA=1 可以强制跳过）。
"""

import glob
import os
import platform

from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))


def rel(pattern):
    return sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, pattern)))


sources = ["csrc/torch_bindings.cpp"] + rel("csrc/cpu/*.cpp")
cuda_sources = rel("csrc/cuda/*.cu") + rel("csrc/cuda/*.cpp")
use_cuda = bool(cuda_sources) and CUDA_HOME is not None and os.environ.get("LLM_INFER_NO_CUDA") != "1"

cxx_flags = ["-O3", "-std=c++20"]
if platform.machine() in ("x86_64", "AMD64"):
    cxx_flags.append("-march=native")

if use_cuda:
    ext = CUDAExtension(
        "llm_infer._C",
        sources + cuda_sources,
        extra_compile_args={"cxx": cxx_flags, "nvcc": ["-O3", "-std=c++20"]},
    )
else:
    ext = CppExtension("llm_infer._C", sources, extra_compile_args={"cxx": cxx_flags})

setup(
    name="llm_infer",
    version="0.1.0",
    packages=find_packages(include=["llm_infer", "llm_infer.*"]),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
