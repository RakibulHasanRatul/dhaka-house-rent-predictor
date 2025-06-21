import os
from setuptools import setup, Extension

os.chdir(os.path.dirname(__file__))

module = Extension(
    "c_impl",
    sources=["python_wrapper.c", "linear_regression_model.c", "matrix_ops.c"],
    include_dirs=["."],
    extra_compile_args=["-O3", "-march=native", "-flto", "-fomit-frame-pointer"],
    extra_link_args=["-flto"],
)


setup(
    name="c_impl",
    version="1.0",
    description="C implementation of linear regression model with pthread",
    ext_modules=[module],
    package_data={"c_impl": ["*.pyi"]},
    include_package_data=True,
)
