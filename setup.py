import sys, os, numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options

# Cython.Compiler.Options.annotate = True

try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)


# scan the directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [".", numpy.get_include()],   # adding the '.' to include_dirs is CRUCIAL!!
        extra_compile_args = ["-O3", "-Wall","-fopenmp"],
        extra_link_args = ['-g','-fopenmp'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        )

# get the list of extensions
packNames = ["potentials"]
packNo = 0
extNames = scandir("potentials")

# and build up the set of Extension objects
extensions = cythonize(
    [makeExtension(name) for name in extNames], 
    compiler_directives={'language_level' : "3"})


setup(
  name="potentials",
  packages=["potentials"],
  ext_modules=extensions,
  cmdclass = {'build_ext': build_ext},
)
