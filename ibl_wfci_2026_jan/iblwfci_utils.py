import masknmf
import os
import sys
import torch
import math
import numpy as np

def load_joao_results(u_path,
                     svt_path,
                     svtcorr_path,
                     frames_avg_path,
                     functional_channel: int = 0):
    """
    Point to 4 files: 
    U.npy (or widefieldU.images.npy if pulling results generated via analysis from alf)
    SVT.npy (or widefieldSVT.uncorrected.npy if pulling results generated via analysis from alf)
    SVTcorr.npy (or widefieldSVT.haemoCorrected.npy if pulling results generated via analysis from alf) 
    frames_average.npy

    This object will return PMD objects describing Joao's pipeline gcamp compression, blood channel compression and hemocorrected results
    """
    u = np.load(u_path)
    v = np.load(svt_path)
    hemocorr_v = np.load(svtcorr_path)
    frame_avg = np.load(frames_avg_path)

    #Convert to torch
    u_sparse = torch.from_numpy(u.reshape((-1, u.shape[2]))).float().to_sparse()
    frame_avg = torch.from_numpy(frame_avg).float()
    v = torch.from_numpy(v).float()
    hemocorr_v = torch.from_numpy(hemocorr_v).float()
    
    v_blood = v[:, 1-functional_channel::2]
    v_gcamp = v[:, functional_channel::2]
    stack_shape = v_blood.shape[1], u.shape[0], u.shape[1]
    
    
    joao_blood = masknmf.PMDArray(stack_shape,
                                u_sparse,
                                v_blood,
                                frame_avg[1-functional_channel, :, :],
                                frame_avg[1-functional_channel, :, :],
                                rescale = True)
    
    joao_gcamp = masknmf.PMDArray(stack_shape,
                                u_sparse,
                                v_gcamp,
                                frame_avg[functional_channel, :, :],
                                frame_avg[functional_channel, :, :],
                                rescale = True)
    
    joao_hemocorr = masknmf.PMDArray(stack_shape,
                                     u_sparse,
                                     hemocorr_v,
                                     frame_avg[functional_channel, :, :],
                                     frame_avg[functional_channel, :, :],
                                     rescale = True)    

    return joao_gcamp, joao_blood, joao_hemocorr

