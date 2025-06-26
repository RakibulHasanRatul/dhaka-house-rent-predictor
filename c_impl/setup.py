import os

from setuptools import Extension, setup

os.chdir(os.path.dirname(__file__))

module = Extension(
    "c_impl",
    sources=["python_wrapper.c", "linear_regression_model.c", "matrix_ops.c"],
    include_dirs=["."],
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-flto",
        "-ffast-math",
        "-fomit-frame-pointer",
        "-funroll-loops",
        "-fno-strict-aliasing",
        "-mavx2",
        "-mfma",
        "-ftree-vectorize",
        "-fdata-sections",
        "-ffunction-sections",
    ],
    extra_link_args=["-flto", "-s", "-Wl,--gc-sections"],
)


setup(
    name="c_impl",
    version="1.0",
    description="C implementation of linear regression model with pthread",
    ext_modules=[module],
    package_data={"c_impl": ["*.pyi"]},
    include_package_data=True,
)
