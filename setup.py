from setuptools import setup, find_packages

setup(
    author = "Alexander Gabriel",
    description = "Package MT @ ITA RWTH - Continual Learning x Composite Prod",
    name = "thesis",
    version = "0.1.2",
    packages = find_packages(include =["thesis", "thesis.*"]),
    dependency_links = [
      "https://data.pyg.org/whl/torch-1.9.0+cu102.html"
    ],
    install_requires = [
      "numpy", 
      "pandas", 
      "torch==1.9.0", 
      "torch_geometric",
      "pyment",
      "augly",
      "Pillow >=8.2.*",
      "umap-learn"
      ],
)

# Manually install !apt-get install libmagic-dev