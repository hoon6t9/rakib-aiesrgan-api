import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name="RealESRGAN",
    py_modules=["RealESRGAN", "RealESRGANplus"],
    version="1.5",
    description="This is an unofficial implementation of Real-ESRGAN and RealESRGANplus in the same module and the implementation process is simplified as much as it can be. Thank you.",
    author="M. Hassan Ibrar",
    url='https://github.com/Hassanibrar632/Real-ESRGAN',
    packages=find_packages(include=['RealESRGAN', 'RealESRGANplus']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)
