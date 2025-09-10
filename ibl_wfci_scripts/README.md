Two experiments: 

1. denoise_then_hemocorr.py: Here we first run PMD+NN on each channel, then run the standard hemocorrection pipeline
2. hemocorr_then_denoise.py: Here we run PMD (no neural net temporal smoothing) on each channel. We then produce an estimate of 
the blood in the gcamp spatial basis. Call this U_gcamp * V_blood_est. 
We then run PMD + NN on the Motion Corrected GCAMP - U_gcamp * V_blood_est.

Note: 
We need to apply some kind of mask to this data. This is currently
computed in a heuristic fashion (by taking the standard deviation + thresholding.)
