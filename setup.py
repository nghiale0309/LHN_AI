from setuptools import setup, Extension
import pybind11
import sys

include_dirs = [
    pybind11.get_include(),
    "include",
    "extern",
]

extra_compile_args = [
    "/std:c++17",
    "/O2",      
    "/fp:fast", 
    "/openmp",
    "/EHsc",    
]

ext_modules = [
    Extension(
        "lhn_AI.core_backend",
        [
            "src/bindings.cpp",
            "src/LinearRegression.cpp",
            "src/LogisticRegression.cpp",
            "src/train_batch_poisson.cpp", 
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

setup(
    name="lhn_AI",
    version="0.1",
    packages=["lhn_AI"],
    ext_modules=ext_modules,
)