# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
    package_list = find_packages()
except ImportError:
    from distutils.core import setup
    package_list = ['pyabad', 'pyabad.data_creation']


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

config = {
    'description': 'Automatic Broadband NIRS Artefact Detection',
    'long_description': readme,
    'author': 'Joshua Russell-Buckland',
    'url': 'https://github.com/buck06191/ABROAD',
    'download_url': 'https://github.com/buck06191/ABROAD/archive/master.zip',
    'author_email': 'joshua.russell-buckland.15@ucl.ac.uk',
    'version': '0.1.0',
    'license': license,
    'install_requires': required,
    'packages': package_list,
    'scripts': [],
    'name': 'ABROAD'
}

setup(**config)
