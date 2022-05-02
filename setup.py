from setuptools import setup, find_packages

setup(
    author = "Alexander Gabriel",
    description = "Package MT @ ITA RWTH - Continual Learning x Composite Prod",
    name = "thesis",
    version = "0.2.1",
    packages = find_packages(include =["thesis", "thesis.*"]),
    dependency_links = [
    ],
    install_requires = [
      "numpy", 
      "pandas", 
      "torch==1.11.*",
      "pyment",
      "augly",
      "Pillow >= 8.2.*",
      "umap-learn",
      "wandb"
      ],
)

# Manually install !apt-get install libmagic-dev
# Always import tensorflow before torchvision in Colab
# -> Tensorflow breaks otherwise...god knows why