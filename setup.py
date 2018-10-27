#!/usr/bin/env python

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py
import os
import subprocess

version = '0.1.0'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('GANTTS_BUILD_VERSION'):
    version = os.getenv('GANTTS_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'gantts', 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


setup(name='gantts',
      version=version,
      description='PyTorch utils for GAN-based text-to-speech synthesis and voice conversion',
      packages=find_packages(),
      cmdclass={
          'build_py': build_py,
          'develop': develop,
      },
      install_requires=[
          "numpy",
          "torch >= 0.4.0",
      ],
      extras_require={
          "train": [
              "docopt",
              "tqdm",
              "tensorboard_logger",
              "nnmnkwii >= 0.0.14",
          ],
          "test": [
              "nnmnkwii >= 0.0.14",
              "nose",
          ],
      })
