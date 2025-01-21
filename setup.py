from setuptools import setup, find_packages

setup(
    name="MultibandMRI",
    version="0.1.0",
    author="Nikolai Mickevicius",
    author_email="nmickevicius@mcw.edu",
    description="Multiband MRI reconstruction",
    long_description="k-space interpolation-based reconstructions of simultaneous multislice MRI data",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)