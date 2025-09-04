from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the consolidated requirements from requirements.txt.  This file
# lists all dependencies needed by neurOS and its Constellation
# extensions.  Using a single source for dependency declarations
# simplifies maintenance and ensures consistency between development
# environments and package installation.  Lines beginning with a
# ``#`` or empty lines are ignored.  If the file is missing during
# editable installs, fall back to a minimal set of core packages.
try:
    with open("requirements.txt", "r", encoding="utf-8") as req_file:
        requirements = [line.strip() for line in req_file if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # fallback minimal requirements for installation without the file
    requirements = [
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "fastapi>=0.103",
        "uvicorn>=0.23",
    ]

setup(
    name="neuros",
    version="2.0.0",
    author="neurOS Development Team",
    description="A modular operating system for brain–computer interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<your-user>/neuros2",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dashboard": ["streamlit>=1.25"],
        "brainflow": ["brainflow>=5.0"],
        "cloud": ["boto3>=1.26", "sagemaker>=2.200"],
        "test": ["pytest>=7.2"],
        "notebook": ["ipykernel>=6.0", "matplotlib>=3.7", "nbformat>=5.7"],
        "constellation": requirements,
    },
    entry_points={
        "console_scripts": [
            "neuros=neuros.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)