These scripts show how to process a single plane of imaging data. Going from a data_raw.bin file to masknmf demixing outputs. 

(1) To run the motion correction pipeline on a data_raw.bin file do: 
```
Some stuff
```
The output of this step will be a motion_corrected.hdf5 file, which contains a dense motion corrected movie and the piecewise rigid shift information

(2) To compress and denoise this motion corrected dataset, do: 
```
python compress_and_denoise.py --bin_file_path /path/to/data.bin --ops_file_path /path/to/ops.npy --out_path /path/to/desired/outputs.hdf5
```

The output of this step will be the standard compression outputs written to a hdf5 file.

(3) Demixing. Demixing takes as input the path to the compression results file (hdf5 file) from step 2, loads the compression arrays runs demixing
```
python demix_data.py --data_path /path/to/compression_results.hdf5 --out_path /path/to/demixing_results.hdf5 --device cuda
```

The output will be a .hdf5 file containing the serialized demixing results.