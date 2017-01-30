try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Automatic Broadband NIRS Artefact Detection',
    'author': 'Joshua Russell-Buckland',
    'url': 'https://github.com/buck06191/pyABAD',
    'download_url': 'https://github.com/buck06191/pyABAD/archive/master.zip',
    'author_email': 'joshua.russell-buckland.15@ucl.ac.uk',
    'version': '0.1.0',
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'pyABAD'
}

setup(**config)
