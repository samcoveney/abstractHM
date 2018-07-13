from setuptools import setup

setup(name='abstracthm',
      version='0.1',
      description='History Matching code in which the user implements the Emulator methods of an abstract base class',
      url='http://github.com/samcoveney/abstractHM',
      author='Sam Coveney',
      author_email='coveney.sam@gmail.com',
      license='GPL-3.0+',
      packages=['abstracthm'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'future',
      ],
      zip_safe=False)
