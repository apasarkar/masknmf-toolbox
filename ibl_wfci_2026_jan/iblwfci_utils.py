import masknmf
import os
import sys
import torch
import math
import numpy as np
from typing import *


class ucla_wf_singlechannel(masknmf.ArrayLike):
    def __init__(self,
                 my_memmap: np.memmap,
                 dtype = np.uint16,
                 channel: int = 0,
                 mask: Optional[np.ndarray] = None,
                 num_frames: Optional[int] = None):
        # print(f"type of my_memmap is {type(my_memmap)}")
        self._channel = channel
        self._dtype = dtype
        self._mmap = my_memmap[:num_frames] if num_frames is not None else my_memmap
        self._shape = (self._mmap.shape[0], self._mmap.shape[2], self._mmap.shape[3])
        self._mask = mask.astype(dtype) if mask is not None else None

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return 3

    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]) -> np.ndarray:
        # return self._mmap[item].copy()
        if isinstance(item, (int, slice, np.ndarray, range)):
            return np.asarray(self._mmap[item, self._channel, :, :]).copy()
        elif isinstance(item, list) or isinstance(item, tuple):
            if len(item) == 1:
                return np.asarray(self._mmap[item[0], self._channel, :, :]).copy()
            elif len(item) == 2:
                return np.asarray(self._mmap[item[0], self._channel, item[1], :]).copy()
            elif len(item) == 3:
                return np.asarray(self._mmap[item[0], self._channel, item[1], item[2]]).copy()


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

