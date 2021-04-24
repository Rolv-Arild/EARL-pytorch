from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='EARL-pytorch',
    version='0.2.0',
    packages=[''],
    url='https://github.com/Rolv-Arild/EARL-pytorch',
    license='MIT License',
    author='Rolv-Arild Braaten',
    author_email='rolv_arild@hotmail.com',
    description='EARL - Extensible Attention-based Rocket League model',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
