import os

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

install_requires = [
    "numpy",
    "pygame"
]

setup(
	name='Arena',
	version='0.0.1',
	description='A Scalable and Configurable Benchmark for Policy Learning',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
	url='https://github.com/Sirui-Xu/Arena',
	author='Sirui Xu',
	author_email='xusirui@pku.edu.cn',
	keywords='',
	license="MIT",
	packages=find_packages(exclude=['example*']),
        include_package_data=False,
        zip_safe=False,
        install_requires=install_requires
)