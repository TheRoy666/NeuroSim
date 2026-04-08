from setuptools import setup, find_packages

setup(
    name="neurosim",
    version="0.1.0-dev",
    author="Ritam",
    description="Physics-first Network Control Theory for in-silico brain stimulation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
    ],
)
