# masknmf-toolbox


[**Installation**](https://github.com/apasarkar/masknmf-toolbox#Installation) |
[**Data Formats**](https://github.com/apasarkar/masknmf-toolbox#examples) |

PyTorch implementation of the masknmf framework for {calcium, voltage, glutamate} imaging analysis. Supports GPU-accelerated:
- Motion Correction
- Compression and Denoising
- Signal Demixing
- High-performance visualization

# Installation

In a conda or python venv, do the following:
```bash

#1. First install the version of PyTorch you want: https://pytorch.org/get-started/locally/

#2. Clone the repo
git clone https://github.com/apasarkar/masknmf-toolbox.git
pip install -e ".[notebook]"
```

# Data Formats
Support currently provided for 
- multipage .tiff files 
- hdf5 files. 

Support for other formats can be easily added by defining a data loader class that implements LazyDataLoader. 