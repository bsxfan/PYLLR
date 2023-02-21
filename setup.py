from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'Python package for log-likelihood-ratio'
LONG_DESCRIPTION = 'Python package for log-likelihood-ratio'

# Setting up
setup(
        name="pyllr", 
        version=VERSION,
        author="Niko Brummer",
        author_email="<niko.brummer@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], 
        
        keywords=['python', 'log likelihood ratio'],
)
