from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="isochem",
    version="1.0.0",
    author="Juan Alday",
    description="1-D photochemical model for planetary atmospheres",
    long_description=long_description,
    long_description_content_type="text/markdown",  # important for Markdown rendering
    url="https://github.com/juanaldayparejo/isochem-dist",
    packages=find_packages(),  #automatically include all subpackages
    install_requires=[
      'numpy',
      'matplotlib',
      'numba>=0.57.0',
      'scipy',
      'h5py',
      'pytest',
    ],
    extras_require={
        'docs': ['sphinx', 'sphinx_rtd_theme'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
