""" Setup and configuration file for cbiotorch. """

import os
from setuptools import setup

__version__ = "0.0.1"


def read(fname):
    """Used to read in README file as long_description."""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="UTF-8").read()


setup(
    name="cbiotorch",
    version=__version__,
    author="Jacob R. Bradley",
    author_email="cobrbradley@gmail.com",
    description=("PyTorch data loaders for cBioPortal datasets."),
    license="MIT",
    keywords="pytorch cbioportal genomics data loaders",
    url="",
    packages=["cbiotorch", "tests"],
    long_description=read("README.md"),
    classifiers=[],
)
