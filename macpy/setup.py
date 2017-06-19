#!/usr/bin/env python

from setuptools import (
    Extension,
    find_packages,
    setup,
)

setup(name='MacPy',
      version='0.0.3',
      description='Python core MAC derivation analytics for bond pricing and bootstrapping curves',
      url='http://git.axiomainc.com:8080/tfs/axiomadev/_git/ContentDev-MAC',
      packages=find_packages('.', include=['macpy', 'macpy.*'],))
