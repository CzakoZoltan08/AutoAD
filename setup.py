from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='autoad',
    version='0.1.0',
    description='Package for Automated Anomaly Detection',
    long_description=readme,
    author='Czako Zoltan',
    author_email='czakozoltan08@gmail.com',
    url='https://github.com/CzakoZoltan08/AutoAD',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
