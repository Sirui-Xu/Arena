import os

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

install_requires = [
    "numpy",
    "Pillow",
    "pygame"
]

setup(
	name='pgle',
	version='0.0.1',
	description='PyGame Graph Based Learning Environment',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
	url='https://github.com/Sirui-Xu/PyGame-Graph-Based-Learning-Environment',
	author='Sirui Xu',
	author_email='xusirui@pku.edu.cn',
	keywords='',
	license="MIT",
	packages=find_packages(),
        include_package_data=False,
        zip_safe=False,
        install_requires=install_requires
)