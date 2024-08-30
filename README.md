# masknmf-toolbox

This toolbox integrates several key tools for end-to-end analysis of neuroimaging data using the masknmf method. The tools included are:

- **Motion correction:** [jnormcorre](https://github.com/apasarkar/jnormcorre)
- **PMD compression and denoising:** [localmd](https://github.com/apasarkar/localmd)
- **Signal Demixing:** [rlocalnmf](https://github.com/apasarkar/rlocalnmf)
- **High-performance scientific plotting and visualization:** [fastplotlib](https://github.com/fastplotlib/fastplotlib)

## Installation

Support is currently only for linux operating systems.

### Step 1: Create a Virtual Environment

First, create and activate a virtual environment using Python 3.11. You can do this with the following command:

```bash
python3.11 -m venv <your_venv_name>
source <your_venv_name>
```

### Step 2: Install system-specific libraries

Based on the system you have (CUDA version, etc.) follow the instructions below to install pytorch, 
jax, and fastplotlib.
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)
- [fastplotlib Installation Guide](https://github.com/fastplotlib/fastplotlib)

### Step 3: Install remaining code

Navigate to the top-level directory of this repository and run

```bash
pip install -e .
```

## Scripts and Interactive Notebooks

We provide scripts and notebooks for the following common use cases:

- Running fused motion correction and denoising
- Running NMF-based signal demixing

## Data Formats
The above scripts only directly support multi-page tiff files. However, the underlying code is highly modular
and is easy to add data-loading support for other input file types. This is only really relevant to the
motion correction steps, since after that step all data is stored in the compressed PMD format in a .npz file.