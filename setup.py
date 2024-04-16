from setuptools import setup, find_packages

from distutils.core import setup as setup_

setup(name='multifil',
      version='0.2',
      description='A spatial half-sarcomere model and the means to run it',
      url='https://github.com/cdw/multifil',
      author='C David Williams',
      author_email='cdave@uw.edu',
      license='MIT',
      packages=find_packages(),
      setup_requires = ['numpy'],
      install_requires=['ujson', 'matplotlib', 'numba', 'scipy'])
