# masknmf-toolbox

This toolbox integrates several key tools for end-to-end analysis of neuroimaging data using the masknmf method. The tools included are:

- **Motion correction:** [jnormcorre](https://github.com/apasarkar/jnormcorre)
- **PMD compression and denoising:** [localmd](https://github.com/apasarkar/localmd)
- **Signal Demixing:** [rlocalnmf](https://github.com/apasarkar/rlocalnmf)
- **High-performance scientific plotting and visualization:** [fastplotlib](https://github.com/fastplotlib/fastplotlib)


## Installation for developers

Support is currently only for linux operating systems. The below instructions are for GPU Systems with
CUDA 12 and python3.11. This is a loose template to follow - 

### Step 1: Create appropriate virtual environments

The initial steps of the pipeline use JAX, and the final demixing step uses PyTorch. These two 
frameworks are difficult to install in a single virtual environment (on GPU), so we use 
two separate python virtual environments, one for registering and compressing the data and another
for demixing and extracting the sources.

```bash
# Make sure you are in the parent directory of this repo

## Make the first venv, installing GPU jax and CPU pytorch
python3.11 -m venv register_and_compress_venv
source register_and_compress_venv/bin/activate

#Modify the below command based on the latest jax install instructions here (https://github.com/jax-ml/jax)
pip install -U "jax[cuda12]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
## Install jnormcorre and localmd from github (latest commit, main branch)
pip install git+https://github.com/apasarkar/jnormcorre.git@main
pip install git+https://github.com/apasarkar/localmd.git@main
pip install -e .
## Install fastplotlib from here: (https://github.com/fastplotlib/fastplotlib)
pip install simplejpeg
pip install -U "fastplotlib[notebook,imgui]"

## Make the second venv, this time installing CPU jax and GPU torch
python3.11 -m venv demixing_venv
source demixing_venv/bin/activate

## Modify the below line based on the latest pytorch install info here: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
pip install jax
## Install localmd and rlocalnmf from github (latest commit, main branch)
pip install git+https://github.com/apasarkar/localmd.git@main
pip install git+https://github.com/apasarkar/rlocalnmf.git@main
pip install -e .
## Install fastplotlib from here: (https://github.com/fastplotlib/fastplotlib)
pip install simplejpeg
pip install -U "fastplotlib[notebook,imgui]"
```


## Data Formats
The above scripts only directly support multi-page tiff files. However, the underlying code is highly modular
and is easy to add data-loading support for other input file types. This is only really relevant to the
motion correction steps, since after that step all data is stored in the compressed PMD format in a .npz file.