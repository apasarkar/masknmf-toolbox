# masknmf-toolbox


[**Installation**](https://github.com/apasarkar/masknmf-toolbox#Installation) |
[**API**](https://github.com/apasarkar/masknmf-toolbox#API) |
[**Data Formats**](https://github.com/apasarkar/masknmf-toolbox#examples) |
[**Paper**](https://github.com/apasarkar/masknmf-toolbox#Paper) |


PyTorch implementation of the masknmf framework for {calcium, voltage, glutamate} imaging analysis. Supports GPU-accelerated:
- Motion Correction
- Compression and Denoising
- Signal Demixing
- High-performance visualization

# Installation
 
Tests are run against Python 3.11 and 3.12, on Linux and Windows using `pip` and `miniforge3`.

## Clone the repository

Until the package is published to PyPI, you will need to clone the repository (or download from GitHub) to install the package.

```bash
git clone https://github.com/apasarkar/masknmf-toolbox.git
cd masknmf-toolbox
```


## pip

Virtual environments are outside the scope of this README, but in general we recommend:
- [UV](https://docs.astral.sh/uv/) (strongly recommended)
- [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments)

```bash
pip install .
```

## miniforge3

The only tested and supported flavor of `anaconda` is [miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers)

```bash
conda create -n masknmf -c conda-forge python=3.12
pip install .
```

## Skip the cloning step

If your environment is already set up, you can skip the cloning step and install directly from a branch of the repository:

```bash
# with standard venvs 
$ pip install git+https://github.com/apasarkar/masknmf-toolbox.git@main
# or with UV
$ uv pip install git+https://github.com/apasarkar/masknmf-toolbox.git@main

Installed 1 package in 0.63ms
 - masknmf-toolbox==0.1.0 (from file:///home/flynn/repos/work/masknmf-toolbox)
 + masknmf-toolbox==0.1.0 (from git+https://github.com/apasarkar/masknmf-toolbox.git@62d3dddfc6e8a024c3ae6284659c871e951ee6c1)
```

## GPU Dependencies

The default installation of PyTorch will not have cuda enabled.
To get the cuda-enabled PyTorch installation

Find which Cuda version you're using (e.g. cuda_12.6)

```bash
nvcc --version
% or
nvidia-smi
```

Windows should have `nvcc` available in the command prompt if you have installed the CUDA toolkit.

If not, you can find it in the CUDA installation directory:

- Windows:`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
- Linux/Unix: `/usr/local/cuda/bin/nvcc`

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:55:00_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```

Install the version of PyTorch that matches your cuda and operating system [on the PyTorch Getting Started](https://pytorch.org/get-started/locally/).

# API
See the notebooks folder for demos on how to use the motion correction, compression, and demixing APIs.

# Data Formats
Support currently provided for 
- multipage .tiff files 
- hdf5 files. 

Support for other formats can be easily added by defining a data loader class that implements LazyDataLoader. 

## Paper

If you use this method, please cite the accompanying [paper](https://www.biorxiv.org/content/10.1101/2023.09.14.557777v1)

> _maskNMF: A denoise-sparsen-detect approach for extracting neural signals from dense imaging data_. (2023). A. Pasarkar\*, I. Kinsella, P. Zhou, M. Wu, D. Pan, J.L. Fan, Z. Wang, L. Abdeladim, D.S. Peterka, H. Adesnik, N. Ji, L. Paninski.
