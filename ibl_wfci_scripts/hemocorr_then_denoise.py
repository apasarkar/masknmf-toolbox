import masknmf
import masknmf
import torch
import os
import wfield
import numpy as np
import fastplotlib as fpl
from typing import *
import cv2
from scipy.ndimage import label
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
from omegaconf import DictConfig, OmegaConf
import hydra

class ucla_wf_singlechannel(masknmf.ArrayLike):
    def __init__(self,
                 my_memmap: np.memmap,
                 dtype = np.uint16,
                 channel: int = 0,
                 mask: Optional[np.ndarray] = None,
                 num_frames: Optional[int] = None):
        self._channel = channel
        self._dtype = dtype
        self._mmap = my_memmap[:num_frames] if num_frames is not None else my_memmap
        frames = self._mmap.shape[0]
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
        if isinstance(item, (int, slice, np.ndarray, range)):
            return np.asarray(self._mmap[item, self._channel, :, :]).copy()
        elif isinstance(item, list) or isinstance(item, tuple):
            if len(item) == 1:
                return np.asarray(self._mmap[item[0], self._channel, :, :]).copy()
            elif len(item) == 2:
                return np.asarray(self._mmap[item[0], self._channel, item[1], :]).copy()
            elif len(item) == 3:
                return np.asarray(self._mmap[item[0], self._channel, item[1], item[2]]).copy()


def _run_pipeline(cfg: DictConfig) -> None:
    data_path = os.path.abspath(cfg.bin_file_path)
    video_obj = wfield.load_stack(data_path, nchannels=2)
    mask = None
    gcamp_channel = ucla_wf_singlechannel(video_obj, channel=0, mask=mask, num_frames=cfg.num_frames_used)
    blood_channel = ucla_wf_singlechannel(video_obj, channel=1, mask=mask, num_frames=cfg.num_frames_used)

    print(f"type {type(video_obj)}")
    print(gcamp_channel.shape)
    pixel_weighting = None #Update this with mask later
    block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

    pmd_gcamp_no_nn = masknmf.compression.pmd_decomposition(gcamp_channel,
                                                            block_sizes,
                                                            gcamp_channel.shape[0],
                                                            max_components=cfg.max_components,
                                                            max_consecutive_failures=cfg.max_consecutive_failures,
                                                            temporal_avg_factor=10,
                                                            spatial_avg_factor=1,
                                                            device=cfg.device,
                                                            temporal_denoiser=None,
                                                            frame_batch_size=1024,
                                                            pixel_weighting=pixel_weighting)

    pmd_hemo_no_nn = masknmf.compression.pmd_decomposition(blood_channel,
                                                           block_sizes,
                                                           blood_channel.shape[0],
                                                           max_components=20,
                                                           max_consecutive_failures=1,
                                                           temporal_avg_factor=1,
                                                           spatial_avg_factor=1,
                                                           device=cfg.device,
                                                           temporal_denoiser=None,
                                                           frame_batch_size=1024,
                                                           pixel_weighting=pixel_weighting)


if __name__ == "__main__":
    config_dict = {
        'bin_file_path': '/path/to/data/frames.bin',
        'mask_file_path': '/path/to/mask/file/',
        'num_frames_used': 60000,
        'out_folder': '/path/to/output/folder/',
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'device': 'cpu',
        'frame_batch_size': 1024,
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    _run_pipeline(cfg)
