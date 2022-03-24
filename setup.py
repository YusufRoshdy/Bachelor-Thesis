from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='IterAnnoy',
    ext_modules=cythonize("IterAnnoy.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)

setup(
    name='CItrAnnoy',
    ext_modules=cythonize("CItrAnnoy.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
