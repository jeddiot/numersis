from setuptools import setup, find_packages

setup(
    name="numersis",  # Package name
    version="0.2.0",  # Professional OOP refactoring
    author="Jedd",
    author_email="yangjedd@gmail.com",
    description="Professional OOP library for numerical analysis algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jeddiot/Numersis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.14.0",
        "scipy>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
)