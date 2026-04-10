from setuptools import setup, find_packages

setup(
    name="backgammon-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "numpy",
        "wandb",
        "h5py",
        "pyyaml",
    ],
    python_requires=">=3.10",
)
