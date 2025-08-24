from setuptools import setup, find_packages

setup(name = 'drnuke-bean',
      version = '0.1',
      description = "Dr Nukes's beancount arsenal",
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      install_requires=[
          'beangulp',
          'beanquery',
      ],
      zip_safe = False)