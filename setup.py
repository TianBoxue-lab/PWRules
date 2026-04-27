# setup.py
from setuptools import setup, find_packages

setup(
    name="pwrules",
    version="1.0.0",
    url="https://github.com/TianBoxue-lab/PWRules",
    packages=find_packages(where="src"),
    package_dir={"": "pwrules"},
    python_requires=">=3.9",
)
