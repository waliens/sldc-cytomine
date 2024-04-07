from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "v2.5.0"

setup(
    name="sldc-cytomine",
    version=__version__,
    description="Cytomine-SLDC, a SLDC binding to Cytomine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["sldc_cytomine"],
    url='https://github.com/waliens/sldc',
    author="Romain Mormont",
    author_email="romain.mormont@gmail.com",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    install_requires=["sldc"]
)
