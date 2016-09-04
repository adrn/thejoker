from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='ebak',
      version='0.1.dev',
      author='adrn',
      author_email='adrn@princeton.edu',
      url='https://github.com/adrn/thejoker',
      license="License :: OSI Approved :: MIT License",
      description='A custom Monte Carlo sampler for the two-body problem.',
      long_description=long_description,
      packages=['thejoker', 'thejoker.celestialmechanics', 'thejoker.sampler'],
)
