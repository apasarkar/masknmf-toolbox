# masknmf command line

## Subcommands

| command | what it does |
|---|---|
| `masknmf` | open the launcher window, which builds and runs a `masknmf run` command |
| `masknmf pipelines` | list the pipelines and their config sections |
| `masknmf params --pipeline P` | list every parameter P accepts and the value it uses by default |
| `masknmf params --pipeline P --json` | print P's default configs as a `--config` file |
| `masknmf run --pipeline P ...` | build P from flags and a `--config` file, and call `P.run(...)` |
| `masknmf run --pipeline P --help` | argparse help showing only P's flags |
| `masknmf view RESULTS.hdf5 ...` | open the viewers for the stages a results file holds |

```bash
masknmf pipelines
masknmf params --pipeline two-photon-calcium
masknmf run --pipeline two-photon-calcium --help
```

Pipelines: `two-photon-calcium`, `one-photon-culture`, `glutamate-calcium-spine`, `widefield-singlechannel`.

## How flags map to the Python API

| Python | CLI |
|---|---|
| `run(data=...)` (the only movie argument) | positional `MOVIE` (optional when `data` accepts `None`) |
| `run(glutamate_channel=..., calcium_channel=...)` (two movies) | `--glutamate-channel PATH --calcium-channel PATH` |
| `run(x: np.ndarray)` | `--x PATH.npy` |
| `run(frame_rate=30)` | `--frame-rate 30` or `--fs 30` |
| `run(scalar=...)` / `__init__(scalar=...)` | `--scalar VALUE` (underscores become dashes) |
| `__init__(motion_correct_config=PiecewiseRigidMotionCorrectionConfig())` | `--motion-correct-kind piecewise-rigid` |
| `__init__(motion_correct_config="skip")` | `--motion-correct-kind skip` |
| any field of any config, including each pass of a multipass config | `--config configs.json` |

Every config argument left out takes the value in the pipeline's `default_configs()`, which `masknmf params` prints.
A `--<section>-kind` naming the default's kind keeps the pipeline's default values; naming another kind gives that
config's own field defaults.

Parsing:
- booleans take `true/false/1/0/yes/no/on/off`: `--remove-intermediates false`
- fields marked `*` in `masknmf params` (arrays, templates, detrenders, pixel/frame weighting) are set from Python only.

Movie input (`MOVIE`, `--glutamate-channel`, `--calcium-channel`, `view --raw`):
- `movie.tif` / `movie.tiff`: `TiffArray`
- a directory: every `.tif`/`.tiff` in it, sorted by name, as a `TiffSeriesLoader` (`TiffArray` if there is only one)
- `movie.h5` / `movie.hdf5`: `Hdf5Array`, which requires `--dataset NAME`

## Config files

`--config FILE` takes json mapping `__init__` argument names to values. Each config is an object whose `"kind"` names
the config (`rigid`, `piecewise-rigid`, `compress`, `compress-denoise`, `multipass`, ...) or the string `"skip"`.
Anything a file leaves out keeps the pipeline's default, so a file only needs what it changes; a section the file
switches to another kind starts from that config's own field defaults instead. A list of passes replaces
the default list: each pass builds on the default pass at the same position, and passes past the end build on the last one.

```bash
# every default, ready to edit
masknmf params --pipeline two-photon-calcium --json > configs.json
masknmf run movie.tif --fs 30 --config configs.json   # the file names the pipeline, so --pipeline can be left out

# rerun with the configs of an earlier run
masknmf run movie.tif --fs 30 --config ./20260923_120000_two-photon-calcium/config.json
```

```json
{
  "pipeline": "TwoPhotonCalciumPipeline",
  "motion_correct_config": {"kind": "piecewise-rigid", "minimum_patch_sizes": [64, 64], "max_deviation_rigid": [3, 3]},
  "compress_config": {"kind": "compress-denoise", "max_components": 30},
  "filtered_demixing_config": {
    "kind": "multipass",
    "DemixingConfigs": [
      {"InitConfig": {"mad_correlation_threshold": 0.7}},
      {},
      {"NMFConfig": {"maxiter": 60}}
    ]
  },
  "device": "cuda"
}
```

Flags win over the file: `--device cpu` replaces the file's device, and a `--<section>-kind` naming another config
than the file's replaces that section with the named config's defaults. A `--<section>-kind` naming the file's own
config keeps the file's values. A run folder's `config.json` holds every
config the run used, with `"*"` for values json cannot hold (arrays, detrenders); `"*"` keeps the default.

---

## two-photon-calcium (`TwoPhotonCalciumPipeline`)

Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise | skip), `spatial-highpass` (spatial-highpass), `filtered-demixing` / `unfiltered-demixing` (multipass: 2 and 3 passes by default).
Run args: `MOVIE` (optional), `--fs` (required), `--exclude-border-radius`, `--remove-intermediates`.
Init args: `--output-folder`, `--frame-batch-size`, `--device {auto,cuda,cpu}`.

Demixing passes without a detrender get the spline detrender the run builds from `--fs`.

```bash
# minimal: all defaults (rigid moco, denoised compression, default demixing) -> ./<timestamp>_two-photon-calcium/results.hdf5
masknmf run --pipeline two-photon-calcium movie.tif --fs 30

# directory of tiffs, run folder under ./session1_out, keep intermediate groups
masknmf run --pipeline two-photon-calcium ./session1_tiffs/ --fs 30 \
    --output-folder ./session1_out \
    --remove-intermediates false

# hdf5 input
masknmf run --pipeline two-photon-calcium raw.h5 --dataset /mov --fs 15

# piecewise-rigid moco at its defaults
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --motion-correct-kind piecewise-rigid

# already registered data: skip moco, zero the border of the shift mask
masknmf run --pipeline two-photon-calcium registered.tif --fs 30 \
    --motion-correct-kind skip --exclude-border-radius 10

# plain PMD compression, tuned, and a stronger spatial highpass
cat > tuned.json <<'EOF'
{
  "compress_config": {"kind": "compress", "block_sizes": [32, 32], "max_components": 30, "frame_range": 5000},
  "spatial_highpass_config": {"filter_sigma": 6}
}
EOF
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --config tuned.json

# resume from an existing run folder's compression, re-running only demixing into the same results.hdf5 (no movie needed)
masknmf run --pipeline two-photon-calcium --fs 30 --compress-kind skip \
    --output-folder ./20260923_120000_two-photon-calcium

# force CPU, smaller batches
masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --device cpu --frame-batch-size 100
```

## one-photon-culture (`OnePhotonCulturePipeline`)

Sections: `motion-correct` (gradient | skip), `compress` (compress | compress-denoise | skip), `demixing` (multipass: 2 passes by default, no detrending).
Run args: `MOVIE`, `--fs` (required), `--indicator-sign {negative,positive}` (required), `--active-frames FILE.npy` (required), `--remove-intermediates`.
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device`.

`--active-frames` is a 1-D `.npy` with one 0/1 entry per frame. It is used as the compression frame weighting.
Gradient motion correction registers to the mean of the first `num_frames_template` frames (default 300).

```bash
# minimal: gradient moco, default compression + demixing
masknmf run --pipeline one-photon-culture voltage.tif --fs 400 \
    --indicator-sign negative --active-frames active.npy

# positive indicator, no moco, data loaded into RAM
masknmf run --pipeline one-photon-culture voltage.tif --fs 1000 \
    --indicator-sign positive --active-frames active.npy \
    --motion-correct-kind skip --load-into-ram true

# a 500 frame template, larger compression blocks, keep the compression group
cat > voltage.json <<'EOF'
{
  "motion_correct_config": {"kind": "gradient", "num_frames_template": 500},
  "compress_config": {"kind": "compress-denoise", "block_sizes": [40, 40], "max_components": 40}
}
EOF
masknmf run --pipeline one-photon-culture voltage.h5 --dataset data --fs 400 \
    --indicator-sign negative --active-frames active.npy --config voltage.json \
    --output-folder ./voltage_out --remove-intermediates false

# re-demix an existing run folder's compression
masknmf run --pipeline one-photon-culture voltage.tif --fs 400 \
    --indicator-sign negative --active-frames active.npy --compress-kind skip \
    --output-folder ./voltage_out/20260923_120000_one-photon-culture
```

## glutamate-calcium-spine (`GlutamateCalciumSpinePipeline`)

Two movie arguments, so both are flags and neither is positional.
Sections: `motion-correct` (rigid), `compress` (compress-denoise), `demixing` (multipass: 2 passes tuned for spines by default).
Run args: `--glutamate-channel`, `--calcium-channel`, `--exclude-initial-frames` (default 200).
Init args: `--output-folder`, `--frame-batch-size`, `--device`.

```bash
# both channels
masknmf run --pipeline glutamate-calcium-spine \
    --glutamate-channel glu.tif --calcium-channel ca.tif

# glutamate only, results in a folder
masknmf run --pipeline glutamate-calcium-spine \
    --glutamate-channel glu.tif --output-folder ./spines_out

# calcium only, keep every frame
masknmf run --pipeline glutamate-calcium-spine \
    --calcium-channel ca.tif --exclude-initial-frames 0

# tune moco and compression; kinds can be left out because each section has one
cat > spines.json <<'EOF'
{
  "motion_correct_config": {"max_shifts": [8, 8]},
  "compress_config": {"block_sizes": [16, 16], "num_epochs": 5}
}
EOF
masknmf run --pipeline glutamate-calcium-spine \
    --glutamate-channel glu.tif --calcium-channel ca.tif --config spines.json
```

## widefield-singlechannel (`WidefieldSinglechannelPipeline`)

Moco and compression only, with no demixing.
Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise, no skip).
Run args: `MOVIE`, `--exclude-border-radius`.
Init args: `--output-folder`, `--frame-batch-size`, `--device`.

```bash
# minimal
masknmf run --pipeline widefield-singlechannel wf.tif

# piecewise-rigid + plain compression
cat > wf.json <<'EOF'
{
  "motion_correct_config": {"kind": "piecewise-rigid", "minimum_patch_sizes": [32, 32]},
  "compress_config": {"kind": "compress", "block_sizes": [16, 16]}
}
EOF
masknmf run --pipeline widefield-singlechannel wf.tif --device cpu --output-folder ./wf_out --config wf.json

# zero a 10 px border of the compression pixel weighting
masknmf run --pipeline widefield-singlechannel wf.tif --exclude-border-radius 10

# no moco, plain compression at its defaults
masknmf run --pipeline widefield-singlechannel wf.tif --motion-correct-kind skip --compress-kind compress
```

## view

```bash
masknmf view results.hdf5 --list                  # print which stage groups the file holds
masknmf view results.hdf5                         # demixing (or compression-only) viewer
masknmf view results.hdf5 --raw movie.tif --fs 30 # + motion and compression viewers (need the raw movie)
masknmf view results.hdf5 --raw raw.h5 --dataset /mov --device cpu
masknmf view results.glutamate.hdf5 --prefix global  # the glutamate pipeline's whole-dendrite result
```
