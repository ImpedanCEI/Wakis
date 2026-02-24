# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

from pathlib import Path

from setuptools import find_packages, setup

# read version
version_file = Path(__file__).parent / "wakis/_version.py"
dd = {}
with open(version_file.absolute(), "r") as fp:
    exec(fp.read(), dd)
__version__ = dd["__version__"]

# read long_description
long_description = (Path(__file__).parent / "README.md").read_text(
    encoding="utf-8"
)

# read requirements.txt for extras_require
with open("requirements.txt") as f:
    notebook_required = f.read().splitlines()

setup(
    name="wakis",
    version=__version__,
    description="3D Electromagnetic Time-Domain wake and impedance solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://wakis.readthedocs.io/",
    author="Elena de la Fuente et al.",
    author_email="elena.de.la.fuente.garcia@cern.ch",
    license="Apache 2.0",
    download_url="https://pypi.python.org/pypi/wakis",
    project_urls={
        "Bug Tracker": "https://github.com/ImpedanCEI/wakis/issues",
        "Documentation": "https://wakis.readthedocs.io/en/latest/index.html",
        "Source Code": "https://github.com/ImpedanCEI/wakis/wakis",
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "pyvista[jupyter]",
        "h5py",
        "tqdm",
    ],
    extras_require={
        "gpu": ["cupy"],
        "all": ["jupyterlab", "ipympl", "bihc", "iddefix"],
        "notebook": notebook_required,
    },
    tests_require=["pytest", "pytest-cov"],
)
