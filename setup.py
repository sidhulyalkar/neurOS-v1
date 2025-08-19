from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuros",
    version="2.0.0",
    author="neurOS Development Team",
    description="A modular operating system for brain–computer interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<your-user>/neuros2",
    # Discover all packages under the root directory. The "neuros" directory
    # contains the top-level package, so we do not want to restrict the search
    # to within "neuros". Using the default find_packages() ensures that
    # setuptools will correctly identify the "neuros" package and any
    # subpackages. The package_dir can remain unspecified when using the
    # default layout.
    packages=find_packages(),
    # Do not override package_dir; leaving it at the default ensures that
    # the "neuros" package is installed properly. Previously we set
    # package_dir={"": "neuros"}, which caused import errors when trying
    # to execute the installed console script because the module could not
    # be found on the Python path.
    # package_dir={"": "neuros"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "fastapi>=0.103",
        "uvicorn>=0.23",
    ],
    extras_require={
        "dashboard": ["streamlit>=1.25"],
        "brainflow": ["brainflow>=5.0"],
        "cloud": ["boto3>=1.26"],
        "test": ["pytest>=7.2"],
        # Extras for Jupyter notebooks and visualisations used in demos
        "notebook": [
            "ipykernel>=6.0",
            "matplotlib>=3.7",
            "nbformat>=5.7",
        ],
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