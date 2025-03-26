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
git clone https://github.com/apasarkar/masknmf-toolbox.git
pip install -e ".[notebook]"
```

# Data Formats
Support currently provided for multipage .tiff files and hdf5 files. Support for other formats can be added by defining a 
data loader class that implements LazyDataLoader. 