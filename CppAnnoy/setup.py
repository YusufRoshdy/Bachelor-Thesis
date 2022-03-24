# python setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='IterAnnoy',
    ext_modules=cythonize(Extension(
        "IterAnnoy",
        sources=['IterAnnoy.pyx'],
        language="c++",
        extra_compile_args=['/openmp'],
        # extra_link_args=['/openmp'],
    )),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
