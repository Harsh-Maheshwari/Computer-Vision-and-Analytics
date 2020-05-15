from distutils.core import setup
from Cython.Build import cythonize
 
setup(name='Motion Detector App',
      ext_modules=cythonize("det.py"))
