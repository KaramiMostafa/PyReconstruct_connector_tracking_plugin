from setuptools import setup, find_packages

setup(
    name="pyrecon-cell-tracker",
    version="0.1.0",
    description="PyReconstruct plugin for DAPI cell tracking",
    author="Mostafa Karami",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.24.1",
        "pandas==2.2.2",
        "scipy==1.14.1",
        "torch>=1.12",
        "scikit-image",
        "roifile",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "pyrecon-track = run_plugin:main",
        ]
    },
)
