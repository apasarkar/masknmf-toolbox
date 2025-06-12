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

Clone the repository

```bash
git clone https://github.com/apasarkar/masknmf-toolbox.git
cd masknmf-toolbox

# if you have a virtual env already set up
pip install .

```
If, for some reason, you're forced to use conda (miniforge3 only)

```bash
conda create -n masknmf -c conda-forge python=3.12
pip install .
```

Install the version of PyTorch that matches your cuda: https://pytorch.org/get-started/locally/

To find this via terminal:

```bash
nvcc --version
% or
nvidia-smi
```

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
