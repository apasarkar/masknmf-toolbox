Usage: 

To run the training loop for training a neural network denoiser: 

```
python train_blindspot_net.py path=/path/to/data/folder/containing/ops_and_bin_files/ outdir=/path/to/output_dir/ device=cuda
```

This will output a file called "neural_net.npz" in the outdir folder.

To compress and denoise datasets using a pre-trained network obtained from the above script: 

```
python compress_and_denoise.py path=/path/to/data/folder/containing/ops_and_bin_files/ outdir=/path/to/output_dir/ device=cuda neural_network=neural_net.npz
```