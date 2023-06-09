from setuptools import setup, find_packages

setup(
    name='tomofast-tools',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    description='Tools for Tomofast-x code',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)