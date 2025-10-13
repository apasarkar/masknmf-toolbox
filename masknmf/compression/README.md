[![CI](https://github.com/Lindsey-cyber/masknmf-toolbox/actions/workflows/test_install.yml/badge.svg)](https://github.com/Lindsey-cyber/masknmf-toolbox/actions)  

**[Examples](#example-usage) | [Inputs](#input-parameters) | [Output](#output) | [APIs](#api-reference-—-pmd_decomposition)**

Next-gen **large-scale functional imaging** compression and denoising using **[Penalized Matrix Decomposition (PMD)](https://arxiv.org/abs/1807.06203)**.

**`pmd_decomposition()`** automatically partitions large functional imaging datasets(e.g., calcium imaging or voltage imaging) into overlapping spatial blocks, performs randomized local SVD for low-rank decomposition, removes noise adaptively, and stitches results into a clean, global representation ready for downstream analysis.

# What can I do with `pmd_decomposition()`?

`pmd_decomposition()` performs a blockwise localized low-rank factorization of your dataset:


$$
\large data \approx U \times V
$$


* **U** — spatial basis matrix (sparse, `[H×W, n_components]`)  
* **V** — temporal basis matrix (dense, `[n_components, T]`)  

Together they represent a compressed and denoised form of your movie.

The process:

1. Split large field of view into overlapping spatial blocks 

2. Perform truncated randomized SVD on each block to capture local low-rank signals

3. Optionally apply temporal and spatial denoisers (from previous neural network training) that smooth the extracted time traces and spatial components

4. Estimates roughness thresholds from simulated noise; discards noisy components and merges valid ones into a global sparse representation (U, V)

5. Outputs a lightweight [PMDArray](#output) object for downstream demixing

# Example Usage

```python
import numpy as np
from masknmf.compression import pmd_decomposition

# Example dataset
data = (np.random.rand(2000, 128, 128) * 255).astype(np.float32)

result = pmd_decomposition(
    dataset=data,
    block_sizes=(16, 16),
    frame_range=150,
    max_components=10,
    device="cuda"
)

U = result.u.to_dense()
V = result.v
print("U:", U.shape, "V:", V.shape)
```

# Inputs

**`dataset`** : `np.ndarray` or `torch.Tensor` or `masknmf.ArrayLike` or `masknmf.LazyFrameLoader`  
Input movie with shape `(T, H, W)`

**`block_sizes`** : `(int, int)`  
Spatial block size used for local decomposition

**`frame_range`** : `int`  
Number of frames to process (subset of total frames)

**`max_components`** : `int`  
Maximum number of components per block

**`device`** : `"cpu"` or `"cuda"`  
Compute device

**`spatial_denoiser, temporal_denoiser`** (optional) : `torch.nn.Module`  
Optional denoiser from previous neural network training

# Output

Returns a `PMDArray` object containing:

- `u` → spatial basis (sparse COO tensor)  
- `v` → temporal basis (dense tensor)  
- `mean_img`, `var_img` → per-pixel statistics  
- `u_local_projector` → (optional) local spatial projector for fast re-projection  

Example reconstruction:

```python
reconstructed = (pmd_result.u.to_dense() @ pmd_result.v).reshape(T, H, W)
```

# Notes for Developers

* This implementation omits explicit spatial/temporal penalties from the original PMD (Buchanan et al., 2018) and instead uses roughness-based thresholds to reject noisy components.

* A randomized SVD algorithm (Halko et al., 2011) is used for efficient local decomposition.

* Each block adaptively selects its rank based on smoothness statistics, stopping when components become too noisy.

* Use .coalesce() to merge duplicate indices in sparse COO tensors.

# Credits

Adapted from Buchanan et al., 2018 (bioRxiv) and used in the maskNMF pipeline (Pasarkar et al., 2023).

Built with PyTorch for flexible CPU/GPU performance.

# API Reference - `pmd_decomposition()` parameters

### 🗂️ Data Input
- **`dataset`** *(masknmf.ArrayLike | masknmf.LazyFrameLoader)*  
  Input dataset of shape `(frames, height, width)`.

- **`block_sizes`** *(Tuple[int, int])*  
  Spatial block size for local decomposition. Each dimension must be ≥10.

- **`frame_range`** *(int)*  
  Number of frames used to estimate spatial and temporal bases.

### ⚙️ Model & Components
- **`max_components`** *(int, default=20)*  
  Maximum number of components per spatial block.

- **`sim_conf`** *(int, default=5)*  
  Percentile value defining roughness thresholds for keeping/rejecting components.

- **`max_consecutive_failures`** *(int, default=1)*  
  Stops accepting new components after this many failures.

### 💾 Performance & Memory
- **`frame_batch_size`** *(int, default=10000)*  
  Max frames loaded into memory at one time.

- **`spatial_avg_factor`** *(int, default=1)*  
  Optional spatial downsampling factor.

- **`temporal_avg_factor`** *(int, default=1)*  
  Optional temporal downsampling factor.

### 🧮 Normalization & Weighting
- **`compute_normalizer`** *(bool, default=True)*  
  If True, estimates per-pixel noise variance (`var_img`).

- **`pixel_weighting`** *(Optional[np.ndarray], default=None)*  
  Optional spatial weighting map `(H, W)` to upweight signal pixels.

### 🧠 Denoisers
- **`spatial_denoiser`** *(Optional[torch.nn.Module], default=None)*  
  Optional callable applied to spatial components.

- **`temporal_denoiser`** *(Optional[torch.nn.Module], default=None)*  
  Optional callable applied to temporal traces.

### 🖥️ Device
- **`device`** *(str, default="cpu")*  
  Compute device: `"cpu"` or `"cuda"`.
