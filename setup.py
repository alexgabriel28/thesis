from setuptools import setup, find_package

setup(
    author = "Alexander Gabriel"
    description = "Master Thesis <> Continual Learning in Composite Production",
    name = "master-thesis",
    version = "0.1.0",
    packages = find_packages(include = ["master-thesis", "master-thesis.*"]),
)