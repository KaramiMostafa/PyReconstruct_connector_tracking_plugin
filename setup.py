from setuptools import setup, find_packages

setup(
    name="pyrecon-cell-tracker",
    version="0.5.0",
    description="PyReconstruct plugins for tracking, segmentation, multiplex RNA mapping, and expert review",
    author="Mostafa Karami",
    packages=find_packages(),
    py_modules=["run_plugin"],
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.24.1",
        "pandas>=2.0",
        "scipy>=1.10",
        "torch>=2.0,<2.6",
        "torchvision>=0.15,<0.21",
        "scikit-image>=0.21",
        "roifile>=2023.0",
        "matplotlib>=3.7",
        "opencv-python-headless>=4.8,<4.12",
        "cellpose>=4.0.4",
        "zarr>=2.16",
        "tifffile>=2024.0",
    ],
    entry_points={
        "console_scripts": [
            "pyrecon-track = run_plugin:main",
        ]
    },
)
