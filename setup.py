import os
import sys
from distutils.core import setup

# import numpy
from setuptools import find_packages

"""
To re-setup: 

    python setup.py sdist bdist_wheel

    pip install -r requirements.txt --process-dependency-links

To test on test pypi:
    
    twine upload --repository testpypi dist/*
    
    # test upload
    pip install -i https://test.pypi.org/simple/ --no-deps eztrack

    twine upload dist/* 
"""

PACKAGE_NAME = "ezinterictal"
version = '0.1'

MINIMUM_PYTHON_VERSION = 3, 6  # Minimum of Python 3.6
REQUIRED_PACKAGES = [
    "numpy>=1.18",
    "scipy>=1.1.0",
    "scikit-learn>=0.24",
    "pandas>=1.0.3",
    "pybids>=0.10",
    "pybv>=0.5.0",
    "joblib>=1.0.0",
    "natsort",
    "tqdm",
    "xlrd",
    "matplotlib>=3.4.0",
    "seaborn",
    "mne>=0.22",
    "mne-bids>=0.7",
]
CLASSIFICATION_OF_PACKAGE = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: Implementation",
    "Natural Language :: English",
]
AUTHORS = [
    "Adam Li",
]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=version,
    author=AUTHORS,
    long_description=open("README.rst").read(),
    packages=find_packages(exclude=["tests"]),
    # include_dirs=[numpy.get_include()],
    install_requires=REQUIRED_PACKAGES,
    classifiers=CLASSIFICATION_OF_PACKAGE,
)
