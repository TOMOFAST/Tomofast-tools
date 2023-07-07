from setuptools import setup, find_packages

setup(
    name='tomofasttools',
    version='1.1.0',
    url='https://github.com/TOMOFAST/Tomofast-tools.git',
    description='Tools for Tomofast-x code',
    packages=find_packages(),
    py_modules=['tomofast_utils'],
    install_requires=['numpy', 'scipy'],
)