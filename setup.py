from setuptools import setup, find_packages

setup(
    name="pyrecon-cell-tracker",
    version="0.1.0",
    description="PyReconstruct plugin for DAPI cell tracking",
    author="Mostafa Karami",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
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
