import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='EARL-pytorch',
    version='0.4.1',
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19",
    ],
    url='https://github.com/Rolv-Arild/EARL-pytorch',
    license='MIT License',
    author='Rolv-Arild Braaten',
    author_email='rolv_arild@hotmail.com',
    description='EARL - Extensible Attention-based Rocket League model',
    # long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
