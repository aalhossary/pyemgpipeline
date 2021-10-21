from distutils.core import setup

from setuptools import find_packages

import os

# Optional project description in README.md:
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
install_requires=['docpie', 'numpy', 'scipy'],
)