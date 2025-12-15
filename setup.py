from setuptools import find_packages, setup

setup(
    name="nki-autotune",
    version="0.1.0-alpha",
    description="NKI Autotune - Kernel optimization tools",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy", "matplotlib", "networkx", "tabulate", "tqdm"],
)
