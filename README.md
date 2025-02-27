# masknmf-toolbox

This toolbox integrates several key tools for end-to-end analysis of neuroimaging data using the masknmf method. The tools included are:

- **Motion correction:** [jnormcorre](https://github.com/apasarkar/jnormcorre)
- **PMD compression and denoising:** [localmd](https://github.com/apasarkar/localmd)
- **Signal Demixing:** [rlocalnmf](https://github.com/apasarkar/rlocalnmf)
- **High-performance scientific plotting and visualization:** [fastplotlib](https://github.com/fastplotlib/fastplotlib)


## Installation for developers

Support is currently only for linux operating systems. The below instructions are for GPU Systems with
CUDA 12 and python3.11. This is a loose template to follow - 

### Step 1: Create appropriate virtual environment
Install dependencies in the same virtual environment

```bash
# Make sure you are in the parent directory of this repo

## Make the first venv, installing GPU jax and CPU pytorch
python -m venv register_and_compress_venv
source register_and_compress_venv/bin/activate

## Install pytorch dependencies for your system (follow official instructions if you want to use GPU here)

## Install jax dependencies for your system (follow official instructions if you want to use GPU here)

git clone https://github.com/apasarkar/masknmf-toolbox.git
cd masknmf-toolbox

##The use of --no-cache-dir should allow you to repeatedly run this code and pull the latest versions of 
## the motion correct, denoising, and demixing libraries
pip install --no-cache-dir -e . 
```

## Data Formats
The above scripts only directly support multi-page tiff files, but you can plug in a 
different dataloader as well. This is only really relevant to the
motion correction steps, since after that step all data is stored in the compressed PMD format in a .npz file.