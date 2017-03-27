from setuptools import setup
from setuptools import find_packages
from keras_contrib import __version__

setup(name='keras_contrib',
      version=__version__,
      description='Keras community contributions',
      author='Fariz Rahman',
      author_email='farizrahman4u@gmail.com',
      url='https://github.com/farizrahman4u/keras-contrib',
      license='MIT',
      install_requires=['keras'],
      packages=find_packages(),
      tests_require=['pytest', 'pytest-pep8', 'pytest-xdist'])
