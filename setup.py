from setuptools import setup, find_packages

#install dependencies from requirements.txt
import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(name='dmdenkf', version='1.0', packages=find_packages(),install_requires=install_requires)
