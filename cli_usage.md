# masknmf command line

## Subcommands

| command | what it does |
|---|---|
| `masknmf` | open the launcher window, which builds and runs a `masknmf run` or `masknmf track` command |
| `masknmf pipelines` | list the pipelines and their config sections |
| `masknmf params --pipeline P` | list every parameter P accepts and the value it uses by default |
| `masknmf params --pipeline P --json` | print P's default configs as a `--config` file |
| `masknmf run --pipeline P ...` | build P from flags and a `--config` file, and call `P.run(...)` |
| `masknmf run --pipeline P --help` | argparse help showing only P's flags |
| `masknmf view RESULTS.hdf5 ...` | open the viewer for the stages a results file holds |
| `masknmf view RESULTS... --classify` | label ROIs across one or more results files or globs |
| `masknmf train-classifier RESULTS... --out NAME` | train a ROI classifier on the saved labels |
| `masknmf classify RESULTS... --classifier F` | classify the ROIs in results files with a trained classifier |
| `masknmf track RESULTS... --out FOLDER` | track ROIs across sessions with ROICaT, saving a tracking folder |
| `masknmf view TRACKING_FOLDER [RESULTS...]` | open the multisession viewer on a ROICaT tracking run |
| `masknmf --version` | print the masknmf version |

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

Without `--output-folder`, the run folder goes next to the movie (inside it when the movie is a directory of tiffs),
or in the working directory when no movie is given.

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

# rerun with the configs of an earlier run, its frame rate included unless --fs is given
masknmf run movie.tif --config ./20260923_120000_two-photon-calcium/config.json

# the same configs on many movies, one run folder beside each; quote the glob so masknmf expands it on every shell
masknmf run "D:/sessions/*/movie.tif" --config ./20260923_120000_two-photon-calcium/config.json
```

Several movies, or a glob, run one after another. A movie that cannot be opened stops the batch before any run starts;
a run that fails is logged and the rest carry on, and the batch ends by listing each movie as done or failed with its
run folder. A config file's `output_folder` is ignored when movies are given: each run folder goes beside its movie,
or under `--output-folder`. From python, `TwoPhotonCalciumPipeline.from_config("<run folder>/config.json")` builds the
same pipeline; pass `frame_rate` and the other run arguments to `run` yourself.

```json
{
  "pipeline": "TwoPhotonCalciumPipeline",
  "configs": {
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
}
```

Flags win over the file: `--device cpu` replaces the file's device, and a `--<section>-kind` naming another config
than the file's replaces that section with the named config's defaults. A `--<section>-kind` naming the file's own
config keeps the file's values. A run folder's `config.json` holds the masknmf version under `masknmf_version`, the
pipeline's class under `pipeline`, the run under `run` (its command line, the device
and gpu it ran on, when it started and finished, its seconds and whether it is running, done or failed), the movie and
array files the run read under `inputs`, each with its resolved `path`, file `name`, `bytes`, `modified` time, `shape`,
`dtype` and, for hdf5, `dataset` by run argument, and every
config the run used under `configs`, with `"*"` for values json cannot hold (arrays, detrenders); `"*"` keeps the default.
Under `timings` it holds each step that ran, with its start time, seconds, whether it finished and, on a cuda device,
its peak cuda memory in GB, updated as the run goes. A step still running after 10 minutes says so in the log, and
every 10 minutes after.
A run that resumes from the folder's compression (`--compress-kind skip --output-folder <run folder>`) keeps the
earlier run's motion correction and compression configs and timings there and replaces the rest.

---

## two-photon-calcium (`TwoPhotonCalciumPipeline`)

Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise | skip), `spatial-highpass` (spatial-highpass), `filtered-demixing` (multipass: 2 passes by default | skip), `unfiltered-demixing` (multipass: 3 passes by default).
Run args: `MOVIE` (optional), `--fs` (required), `--exclude-border-radius`, `--remove-intermediates`.
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device {auto,cuda,cpu}`, `--log-level {debug,info,warning}`.

`--load-into-ram true` reads the whole raw movie into RAM before motion correction, so every later pass (moco, compression, the one-photon raw regression) reads memory instead of the file. It needs about the movie's size in free RAM. The glutamate pipeline always does this and has no flag.

Demixing passes without a detrender get the spline detrender the run builds from `--fs`.
`--filtered-demixing-kind skip` ends the run after compression; a later `--compress-kind skip` run demixes it.

```bash
# minimal: all defaults (rigid moco, denoised compression, default demixing) -> <movie's folder>/<timestamp>_two-photon-calcium/results.hdf5
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
  "configs": {
    "compress_config": {"kind": "compress", "block_sizes": [32, 32], "max_components": 30, "frame_range": 5000},
    "spatial_highpass_config": {"filter_sigma": 6}
  }
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
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device`, `--log-level`.

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
  "configs": {
    "motion_correct_config": {"kind": "gradient", "num_frames_template": 500},
    "compress_config": {"kind": "compress-denoise", "block_sizes": [40, 40], "max_components": 40}
  }
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

Two movie arguments, so both are flags and neither is positional. Either one can be left out, though `masknmf params`
lists both as required. The run folder holds `results.glutamate.hdf5` and `results.calcium.hdf5` instead of `results.hdf5`.
Sections: `motion-correct` (rigid), `compress` (compress-denoise), `demixing` (multipass: 2 passes tuned for spines by default).
Run args: `--glutamate-channel`, `--calcium-channel`, `--exclude-initial-frames` (default 200).
Init args: `--output-folder`, `--frame-batch-size`, `--device`, `--log-level`.

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
  "configs": {
    "motion_correct_config": {"max_shifts": [8, 8]},
    "compress_config": {"block_sizes": [16, 16], "num_epochs": 5}
  }
}
EOF
masknmf run --pipeline glutamate-calcium-spine \
    --glutamate-channel glu.tif --calcium-channel ca.tif --config spines.json
```

## widefield-singlechannel (`WidefieldSinglechannelPipeline`)

Moco and compression only, with no demixing.
Sections: `motion-correct` (rigid | piecewise-rigid | skip), `compress` (compress | compress-denoise, no skip).
Run args: `MOVIE`, `--exclude-border-radius`.
Init args: `--output-folder`, `--load-into-ram`, `--frame-batch-size`, `--device`, `--log-level`.

```bash
# minimal
masknmf run --pipeline widefield-singlechannel wf.tif

# piecewise-rigid + plain compression
cat > wf.json <<'EOF'
{
  "configs": {
    "motion_correct_config": {"kind": "piecewise-rigid", "minimum_patch_sizes": [32, 32]},
    "compress_config": {"kind": "compress", "block_sizes": [16, 16]}
  }
}
EOF
masknmf run --pipeline widefield-singlechannel wf.tif --device cpu --output-folder ./wf_out --config wf.json

# zero a 10 px border of the compression pixel weighting
masknmf run --pipeline widefield-singlechannel wf.tif --exclude-border-radius 10

# no moco, plain compression at its defaults
masknmf run --pipeline widefield-singlechannel wf.tif --motion-correct-kind skip --compress-kind compress
```

## view

`RESULTS` is one or more `.h5`/`.hdf5` files or globs, quoted so masknmf expands them on every shell; `**` walks every
folder below. Globs never enter `.zarr` stores and leave out `.labels.hdf5` sidecars and any results file a newer
curated file replaces ([curated results files](#curated-results-files)). The demixing viewer opens one file; several
need `--classify`.

```bash
masknmf view "sessions/*/results.hdf5" --list      # print which stage groups each file holds
masknmf view results.hdf5                          # demixing viewer (compression only when there is no demixing)
masknmf view results.hdf5 --raw movie.tif --fs 30  # + raw and registered panels; a registration-only file needs --raw
masknmf view results.hdf5 --raw movie.tif --compression  # + lag-1 autocorrelation images of the registered, compressed and residual movies
masknmf view results.hdf5 --raw raw.h5 --dataset /mov --device cpu
masknmf view results.glutamate.hdf5 --prefix global  # the glutamate pipeline's whole-dendrite result
```

## classification

Label ROIs by hand, train a ROICaT classifier on the labels, then classify new sessions. Each results file with
top-level demixing results is one session; files without them are skipped. Labels and predictions are saved beside
each results file, `results.hdf5 -> results.labels.hdf5`.

```bash
# label ROIs across sessions; --classifier is where Train saves, and an existing file is selected for Classify
masknmf view "sessions/*/results.hdf5" --classify --labels soma,dendrite,junk --classifier cells.roicat_classifier

# train on the labels; every ROI in every session must be labeled. writes cells.roicat_classifier and cells.training.json
masknmf train-classifier "sessions/*/results.hdf5" --out cells

# classify new sessions; unlabeled ROIs take the predictions
masknmf classify "new_sessions/**/results.hdf5" --classifier cells.roicat_classifier --device cpu
```

`--classify` reads only the top-level demixing results, so it takes neither `--prefix`, `--raw` nor `--compression`.

## multisession

`masknmf track` matches ROIs across sessions with ROICaT, one session per results file, numbered in the order given
(a glob sorts by path). Each file needs top-level demixing results; files without them are skipped, and at least two
sessions must remain.

```bash
# track three days; --um-per-pixel is the imaging resolution (default 1.2), --device cpu keeps ROICaT off the gpu
masknmf track day1/results.hdf5 day2/results.hdf5 day3/results.hdf5 --out ./tracking
masknmf track "sessions/day*/*/results*.hdf5" --out ./tracking --um-per-pixel 0.8 --device cpu
```

The `--out` folder holds `<timestamp>.tracking.results_all.richfile.zip`, `<timestamp>.tracking.run_data.richfile.zip`,
`<timestamp>.tracking.params.yaml` and `<timestamp>.tracking.masknmf_sessions.json`, the results file of each session
as the run saw it. The launcher's `tracking` entry builds the same command. From Python:
`RoicatTracker().run_tracking(RoicatDataAdapter.from_masknmf(files))` then `.to_roicat_dir(folder)`.

`masknmf view` opens a tracking folder in the multisession viewer:

```bash
# the sessions' results files where the tracking run recorded them
masknmf view ./tracking

# the results files moved (another machine, a shared drive): pass them after the folder, as paths or a glob
masknmf view ./tracking "sessions/day*/*/results.hdf5"

# print the sessions and the file each one reads, marking missing files, without opening the viewer
masknmf view ./tracking "sessions/day*/*/results.hdf5" --list
```

Results files given after the folder replace the recorded ones. Each goes to the session with its ROI count, so a glob
can match them in any order; files already in session order stay as given, and two sessions with the same ROI count need
the files in session order. A file whose ROI count differs from its session's is refused: it was curated or re-run after
tracking ([curated results files](#curated-results-files)). Of the view flags, a tracking folder uses only `--device`
and `--list`.

Keep the `.richfile.zip` files zipped. Unzipping `run_data` on Windows silently drops every file past the 260 character
path limit and the folder no longer loads; a zip beside its unzipped copy is refused as two results.

The viewer shows up to three sessions side by side. The Sessions tab picks which session each panel shows and which
sessions' traces are drawn; left / right move every panel one session and `[` / `]` one page, and double-clicking a
footprint selects its cluster in every session. The Clusters table lists each cluster,
the number of sessions it was found in, its similarity and silhouette, and its ROI in each session on screen.

## curated results files

The demixing viewer's Demix (Python: `SingleSessionDemixingVis(..., results_path=...)`, or
`masknmf.demixing.update_signals` then `write_curated`) never changes a results file. It writes a new one beside it,
named after the results file it descends from:

| file | holds |
|---|---|
| `results.hdf5` | the run: registration, compression and demixing groups |
| `results.<yyyy-mm-dd-HH-MM-SS>.curated.hdf5` | only `DemixingResults`, after the full NMF pass; its `description` attribute says which signals were removed (and by which filter) and how many drawn ROIs were added |
| `results.<yyyy-mm-dd-HH-MM-SS>.curated.labels.hdf5` | the curated file's own labels; the parent's are not copied, curation renumbers the ROIs |

Curating a curated file writes another `results.<later stamp>.curated.hdf5`; the stem stays `results`. The glutamate
pipeline's files give `results.calcium.<stamp>.curated.hdf5` and `results.glutamate.<stamp>.curated.hdf5`. Curated
files from before 2026-09-30 are named `<stamp>.curated.hdf5`; rename them to `results.<stamp>.curated.hdf5` so they
are grouped with their results file.

Which file a command reads:

| RESULTS given as | file used |
|---|---|
| a file path | exactly that file |
| a glob (CLI), or a folder (classification viewer's Open) | for each results file, its newest curated file when the glob matched one, else the results file; the CLI prints how many it left out |

```bash
masknmf view "sessions/*/*.hdf5" --classify              # one file per run: the newest curated one, else results.hdf5
masknmf view "sessions/*/results.hdf5" --classify        # the uncurated results only: the glob matches no curated file
masknmf view sessions/day1/results.2026-09-30-12-27-07.curated.hdf5   # one given version
```

Per command:

- `masknmf view FILE`: the demixing viewer on that file. A curated file holds no registration group, so there are no
  shift traces or registered panel; `--raw` still adds the raw panel. Demix again writes the next curated file.
- `masknmf view ... --classify`, `train-classifier`, `classify`: each file used is one session, labeled in its own
  `.labels.hdf5`; a curated file starts unlabeled.
- `masknmf track` tracks the files used, and its clusters index each file's ROIs by position, so a
  tracking belongs to those files. Curate every day first, then track; a day curated later needs tracking again.
- `masknmf view TRACKING_FOLDER [FILES...]`: the multisession viewer on the files the tracking recorded; files after
  the folder replace them, each placed at the session with its ROI count. A file whose ROI count differs from the tracking's is refused
  (`session 1: ... holds 60 ROIs, the tracking 63; the file was curated or re-run after tracking, so track the
  sessions' current files again`).

Python:

| call | does |
|---|---|
| `masknmf.demixing.write_curated(path, results, drop, num_masks, filters)` | writes `<stem>.<stamp>.curated.hdf5` beside `path`, returns its path |
| `masknmf.demixing.results_stem(path)` | the results file a file is or descends from: `results.calcium.<stamp>.curated.hdf5 -> results.calcium` |
| `masknmf.demixing.latest_results(paths)` | the glob rule above: one file per results file, its newest curated one when present |
| `RoicatTrackingResults(..., session_files=...)`, `.session_files = ...`, `from_roicat_dir(..., session_files=...)` | refuse a file whose ROI count differs from the tracking's |
