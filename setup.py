from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="twistHS",
    version="1.2.0",
    description="A package for twisted heterostructure generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wei Li",
    url="https://github.com/i-wli/twistHS",
    python_requires=">=3.12",
    packages=['twistHS', 'twistHS.lib'],
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "ase>=3.22.1",
        "scipy>=1.10.0",
        "gradio>=4.0.0",
        "pytest>=7.0.0",  # for running tests
    ],
    entry_points={
        'console_scripts': [
            'twist_cli=twistHS.twist_cli:main',
            'twist_gui=twistHS.twist_gui:main',
        ],
    },
)