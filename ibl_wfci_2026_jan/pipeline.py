import masknmf
import torch
import os
import sys
from typing import *
import argparse
from scipy.interpolate import interp1d

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
        return self._mmap[item].copy()


def per_pixel_lstsq(u_gcamp: torch.sparse_coo_tensor,
                    u_hemo: torch.sparse_coo_tensor,
                    v_signal: torch.tensor,
                    v_hemo_interp: torch.tensor,
                    batch_size: int = 600,
                    device="cpu"):
    u_gcamp = u_gcamp.to(device)
    u_hemo = u_hemo.to(device)
    v_hemo_interp = v_hemo_interp.to(device)
    v_signal = v_signal.to(device)
    num_pixels = u_gcamp.shape[0]
    num_iters = math.ceil(u_gcamp.shape[0] / batch_size)

    outputs = torch.zeros(num_pixels, dtype=u_gcamp.dtype, device=device)
    for k in tqdm(range(num_iters)):
        start_elt = batch_size * k
        end_elt = min(num_pixels, start_elt + batch_size)
        i_select = torch.arange(start_elt, end_elt, device=device).long()
        u_subset_gcamp = torch.index_select(u_gcamp, 0, i_select).coalesce()
        u_subset_hemo = torch.index_select(u_hemo, 0, i_select).coalesce()
        signal = torch.sparse.mm(u_subset_gcamp, v_signal)
        blood = torch.sparse.mm(u_subset_hemo, v_hemo_interp)

        outputs[start_elt:end_elt] = torch.sum(signal * blood, dim=1) / (1e-6 + torch.sum(blood * blood, dim=1))

    return outputs

def compute_blood_basis(u_gcamp: torch.sparse_coo_tensor,
                        u_projector_gcamp: torch.sparse_coo_tensor,
                        u_hemo: torch.sparse_coo_tensor,
                        v_hemo_interp: torch.tensor,
                        coefficients: torch.tensor,
                        device="cpu"):
    u_gcamp = u_gcamp.to(device)
    u_project_gcamp = u_projector_gcamp.to(device)
    u_hemo = u_hemo.to(device)
    v_hemo_interp = v_hemo_interp.to(device)
    cofficients = coefficients.to(device)
    indices = torch.arange(len(coefficients)).unsqueeze(0).repeat(2, 1)  # [[0,1,2..],[0,1,2..]]
    size = (len(coefficients), len(coefficients))
    sparse_diag = torch.sparse_coo_tensor(indices.to(device), coefficients.to(device), size).to(device)

    uts = torch.sparse.mm(u_projector_gcamp.t().to(device), sparse_diag)
    utsu = torch.sparse.mm(uts, u_hemo)

    utsuv = torch.sparse.mm(utsu, v_hemo_interp)

    return utsuv

def hemocorr_basis(blood_basis,
                   gcamp_basis,
                   device='cpu'):
    return gcamp_basis.to(device) - blood_basis.to(device)

def hemocorr_pipeline(bin_file_path: str,
                      mask_file_path: str,
                      out_folder: str,
                      num_frames_used: Optional[int] = None,
                      block_size_dim1: int=100,
                      block_size_dim2: int=100,
                      max_components: int=20,
                      max_consecutive_failures: int=1,
                      spatial_avg_factor: int=1,
                      temporal_avg_factor: int=1,
                      frame_batch_size: int=1000):

    out_save_dir = os.path.abspath(out_folder)
    data_path = os.path.abspath(bin_file_path)
    video_obj = wfield.load_stack(data_path, nchannels=2)
    mask = np.load(os.path.abspath(mask_file_path)).astype('float')

    if num_frames_used is None:
        num_frames_used = video_obj.shape[0]

    gcamp_channel = ucla_wf_singlechannel(video_obj,
                                          channel=0,
                                          mask=mask,
                                          num_frames=num_frames_used)
    blood_channel = ucla_wf_singlechannel(video_obj,
                                          channel=1,
                                          mask=mask,
                                          num_frames=num_frames_used)

    block_sizes = [block_size_dim1, block_size_dim2]
    gcamp_strat = masknmf.CompressDenoiseStrategy(gcamp_channel,
                                                  block_sizes=block_sizes,
                                                  max_components=max_components,
                                                  max_consecutive_failures=max_consecutive_failures,
                                                  spatial_avg_factor=spatial_avg_factor,
                                                  temporal_avg_factor=temporal_avg_factor,
                                                  pixel_weighting=mask,
                                                  noise_variance_quantile=0.3,
                                                  num_epochs=10,
                                                  frame_batch_size=frame_batch_size)

    pmd_gcamp = gcamp_strat.compress()

    blood_strat = masknmf.CompressDenoiseStrategy(blood_channel,
                                                  block_sizes=block_sizes,
                                                  max_components=max_components,
                                                  max_consecutive_failures=max_consecutive_failures,
                                                  spatial_avg_factor=spatial_avg_factor,
                                                  temporal_avg_factor=temporal_avg_factor,
                                                  pixel_weighting=mask,
                                                  noise_variance_quantile=0.3,
                                                  num_epochs=10,
                                                  frame_batch_size=frame_batch_size)

    pmd_blood = blood_strat.compress()

    blood_indices = np.array([i*2+1 for i in range(pmd_blood.shape[0])])
    gcamp_indices = np.arrar([i*2 for i in range(pmd_gcamp.shape[0])])

    v_hemo_interp = interp1d(blood_indices,
                             pmd_blood.v.cpu().numpy(),
                             axis=1,
                             fill_value='extrapolate')(gcamp_indices)

    v_hemo_interp = torch.from_numpy(v_hemo_interp).to(pmd_blood.device)

    regression_coefficients = per_pixel_lstsq(pmd_gcamp.u,
                                              pmd_blood.u,
                                              pmd_gcamp.v,
                                              v_hemo_interp.v,
                                              batch_size=300,
                                              device='cuda')

    blood_basis = compute_blood_basis(pmd_gcamp.u,
                                      pmd_gcamp.u_local_projector,
                                      pmd_hemo.u,
                                      v_hemo_interp,
                                      regression_coefficients,
                                      device='cpu')

    v_corr = hemocorr_basis(blood_basis,
                            pmd_gcamp.v,
                            device='cuda')

    hemo_corr_est = masknmf.PMDArray(pmd_gcamp.shape,
                                     pmd_gcamp.u.cpu(),
                                     v_corr.cpu(),
                                     pmd_gcamp.mean_img.cpu(),
                                     pmd_gcamp.var_img.cpu(),
                                     u_local_projector=pmd_gcamp.u_local_projector.cpu(),
                                     device='cpu')

    ## Save results:
    gcamp_path = os.path.join(out_save_dir, 'gcamp.hdf5')
    blood_path = os.path.join(out_save_dir, 'blood.hdf5')
    hemocorr_path = os.path.join(out_save_dir, 'hemocorr.hdf5')

    pmd_gcamp.export(gcamp_path)
    pmd_blood.export(blood_path)
    hemo_corr_est.export(hemocorr_path)

if __name__ == "__main__":
    config_dict = {
        'bin_file_path': '/path/to/data/frames.bin',
        'mask_file_path': '/path/to/mask/file/',
        'num_frames_used': None,
        'out_folder': '/path/to/output/folder/',
        'block_size_dim1': 100,
        'block_size_dim2': 100,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'frame_batch_size': 1024,
    }

    parser = argparse.ArgumentParser()
    for key in config_dict.keys():
        curr_key = "--"+key
        parser.add_argument("--"+key)

    args = vars(parser.parse_args())
    #Delete none values.
    print(args)
    args = {key:val for key, val in args.items() if val is not None}
    final_inputs = {**config_dict, **args}

    hemocorr_pipeline(**final_inputs)


