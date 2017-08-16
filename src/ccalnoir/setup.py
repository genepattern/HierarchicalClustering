from setuptools import setup
from setuptools.command.install import install


class InstallCommand(install):
    def run(self):
        install.run(self)
        from distutils import log
        log.set_verbosity(log.DEBUG)


setup(name='ccalnoir',
      description='Computational Cancer Analysis Library -- sans R',
      packages=['ccalnoir'],
      version='0.0.2',
      author='Huwate (Kwat) Yeerna (Medetgul-Ernar); Modifier: Edwin Juarez',
      author_email='kwat.medetgul.ernar@gmail.com, ejuarez@ucsd.edu',
      license='MIT',
      url='https://github.com/ucsd-ccal/ccal',
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.5'],
      keywords=['computational cancer biology genomics'],
      install_requires=[
          # 'rpy2', ## Commented-out by EJ on 2017-07-20
          'biopython',
          'plotly',
      ],
      cmdclass={'install': InstallCommand},
      package_data={})
