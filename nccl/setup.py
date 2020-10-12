import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "py_nccl_sendrecv",
        sources=["_nccl.pyx"],
        include_dirs=[os.path.abspath("/usr/local/cuda/include/")],
        library_dirs=[os.path.abspath("/usr/local/cuda/lib/")],
        libraries=["nccl", "cudart"],
        language="c++",
    )
]

setup(name="py_nccl_sendrecv",
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}))
