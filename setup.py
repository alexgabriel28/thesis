from setuptools import setup, find_packages

setup(
    author = "Alexander Gabriel",
    description = "Package MT @ ITA RWTH - Continual Learning x Composite Prod",
    name = "thesis",
    version = "0.1.1",
    packages = find_packages(include =["master-thesis", "master-thesis.*"]),
    install_require = [
      "numpy", 
      "pandas", 
      "torch==1.9.*", 
      "torch_geometric",
      "torchvision",
      "pyment",
      "augly",
      "PIL"
      ]
)