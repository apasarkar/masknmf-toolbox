# GlutamateCalciumSpinePipeline

```python
import tifffile
from masknmf.pipelines.subcellular import GlutamateCalciumSpinePipeline

glu = tifffile.imread("green.tif")
calcium = tifffile.imread("red.tif")

pipe = GlutamateCalciumSpinePipeline(output_folder="results/")
pipe.run(glutamate_channel=glu, calcium_channel=calcium)
```

Writes `results/<YYYYMMDD_HHMMSS>_glutamate_calcium_spine_results/`. Either channel may be `None`.
`run` drops the first 200 frames (`exclude_initial_frames`); every output has `frames - 200` frames.

```python
from pathlib import Path
import numpy as np
import masknmf

run = Path("results/20260908_112158_glutamate_calcium_spine_results")
calcium = tifffile.imread("red.tif")[200:]
timings = np.arange(calcium.shape[0]) / 19.66  # optional seconds axis
```

## 1. Motion correction

| file | hdf5 groups |
|---|---|
| `calcium_moco.hdf5`, `glutamate_moco.hdf5` | `RigidRegistrationArray` (shifts, sinc_margin), `RigidMotionCorrector` (template, max_shifts, batch_size) |

The movie is not stored. Pass the cropped raw stack back in.

```python
reg = masknmf.RigidRegistrationArray.from_hdf5(run / "calcium_moco.hdf5", input_movie=calcium)
```

| attribute | shape / type |
|---|---|
| `reg.shifts` | `(frames, 2)` tensor, `(dy, dx)` px |
| `reg.strategy.template` | `(H, W)` ndarray |
| `reg[i]` | `(1, H, W)` tensor, registered frame |
| `reg[:]` | `(frames, H, W)` tensor |

```python
masknmf.MotionCorrectionVis(reg, frame_timings=timings, mean_subtract=True).show()
```

## 2. Compression

| file | hdf5 groups |
|---|---|
| `pmd_calcium.hdf5`, `pmd_glutamate.hdf5` | `PMDArray` (u, v, mean_img, var_img, u_local_projector, shape) |

```python
pmd = masknmf.PMDArray.from_hdf5(run / "pmd_calcium.hdf5")
```

| attribute | shape / type |
|---|---|
| `pmd.shape` | `(frames, H, W)` |
| `pmd.u`, `pmd.v` | `(H*W, rank)` sparse, `(rank, frames)` |
| `pmd.mean_img`, `pmd.var_img` | `(H, W)` tensors |
| `pmd[i]` | `(1, H, W)` ndarray, raw scale |

Compression ran on the registered movie masked to the dendrite (`mean_img == 0` outside).

```python
moco = reg[:].cpu().numpy()
masknmf.CompressionVis(moco, pmd, frame_timings=timings, device="cuda").show()
```

## 3. Demixing

| file | contents |
|---|---|
| `glutamate_spine_demixing.hdf5` | spines found in glutamate |
| `calcium_spine_demixing.hdf5` | same footprints, traces refit on calcium |
| `glutamate_global_activity_demixing.hdf5`, `calcium_global_activity_demixing.hdf5` | one whole-dendrite component |

All are `DemixingResults` (hdf5 group `DemixingResults`).

```python
res = masknmf.DemixingResults.from_hdf5(run / "glutamate_spine_demixing.hdf5", device="cuda")
```

| attribute | shape / type |
|---|---|
| `res.a` | `(H*W, K)` sparse footprints |
| `res.ac_array.export_a()` | `(H, W, K)` ndarray |
| `res.c` | `(frames, K)` traces |
| `res.mean_img` | `(H, W)` |
| `res.pmd_array`, `res.ac_array`, `res.residual_array` | lazy `(frames, H, W)` movies |

```python
masknmf.SingleSessionDemixingVis(res, frame_timings=timings, device="cuda").show()
```

## Scripts

`fpl.loop.run()` after `show()` outside notebooks. `examples/pipelines/run_glu_ca_pipeline.py` runs,
`examples/pipelines/view_glu_ca_results.py` views.
