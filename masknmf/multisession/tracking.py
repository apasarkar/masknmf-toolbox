from masknmf.demixing.demixing_results import DemixingResults
import roicat
from roicat.data_importing import Data_roicat
import torch
import os
import sys
from typing import *
import masknmf
from pathlib import Path
import roicat
from roicat.data_importing import Data_roicat
import numpy as np
import scipy
import scipy.sparse


class DemixingRoicat(Data_roicat):
    _notes = "um per pixel always 1.2 for IBL 2p mesoscope data"

    def __init__(self,
                 mean_img_list: List[np.ndarray],
                 spatial_fp_list: List[scipy.sparse.coo_matrix],
                 um_per_pixel: float = 1.2,
                 roi_image_dims: tuple[int, int] = (36, 36),
                 highpass_sigma: Optional[int] = 3):
        """
        Generic interface for doing multi-session tracking with any analysis pipeline
        Args:
            mean_img_list (List[np.ndarray]): List of mean images from each imaging session. Each image should have same dimensions.
            spatial_fp_list (List[np.ndarray]): List of spatial footprint arrays, one for each session. Each individual array has shape (num_rois, num_pixels).
                Each spatial footprint is flattened into a row of this array in "C" order.
            um_per_pixel (float): Describes the resolution of the imaging
            roi_image_dims (tuple[int, int]): Each ROI is spatially cropped for purposes of feature extraction in the ROICat pipeline. This specifies the crop dimensions.
            highpass_sigma (int): We highpass filter the mean image to define an "enhanced" mean image (this is what s2p does) for use in the tracking pipeline.
        """

        super().__init__()
        self.um_per_pixel = um_per_pixel
        self._highpass_sigma = highpass_sigma
        self._mean_img_list = mean_img_list
        self.set_FOVHeightWidth(int(mean_img_list[0].shape[0]), int(mean_img_list[1].shape[1]))
        self.set_fov_imgs_from_mean_imgs()
        self.set_spatialFootprints(spatial_fp_list, self.um_per_pixel)
        self.transform_spatialFootprints_to_ROIImages(out_height_width=roi_image_dims)

    def set_fov_imgs_from_mean_imgs(self):
        fov_list = self._filter_and_normalize_mean_img()
        return self.set_FOV_images(fov_list)

    def _filter_and_normalize_mean_img(self):
        """
        This pipeline convolves each image with a
        """
        if self._highpass_sigma is None:
            return self._mean_img_list
        else:
            """
            Spatially high-pass filter each image and normalize the data between 0 and 1
            """
            radius = int(torch.ceil(torch.tensor(2 * self._highpass_sigma)).item())
            size = 2 * radius + 1
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
            yy, xx = torch.meshgrid(coords, coords, indexing='ij')

            # 2D Gaussian
            kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * self._highpass_sigma ** 2))

            # Normalize so sum = 1
            kernel /= kernel.sum()
            kernel *= -1

            kernel[radius, radius] += 1

            # Reshape for conv2d: (out_ch, in_ch, H, W)
            kernel = kernel.unsqueeze(0).unsqueeze(0)

            new_list = []
            for k in range(len(self._mean_img_list)):
                curr_mean_img = torch.from_numpy(self._mean_img_list[k]).float()

                image = curr_mean_img.unsqueeze(0).unsqueeze(0)
                image = torch.nn.functional.pad(image,
                                                pad=(radius, radius, radius, radius), mode="reflect")

                # Convolve
                output = torch.nn.functional.conv2d(image, kernel, padding=0).squeeze(0).squeeze(0).cpu()

                # Normalize + clip
                p1 = torch.quantile(output, 0.01)
                p99 = torch.quantile(output, 0.99)
                x_clipped = torch.clamp(output, min=p1, max=p99)
                x_norm = (x_clipped - p1) / (p99 - p1)

                x_norm = x_norm.numpy()
                new_list.append(x_norm)
            return new_list

    @classmethod
    def from_masknmf(cls,
                     demixing_result_files: list[str | Path],
                     **kwargs):
        dmr_list = []
        spatial_footprint_list = []
        mean_img_list = []
        for fname in demixing_result_files:
            dmr = masknmf.DemixingResults.from_hdf5(fname)
            footprint = extract_masknmf_spatial_footprints(dmr)
            mean_img = extract_masknmf_mean_img(dmr)
            spatial_footprint_list.append(footprint)
            mean_img_list.append(mean_img)

        return cls(mean_img_list,
                   spatial_footprint_list,
                   **kwargs)

    @classmethod
    def _from_suite2p(cls,
                      ops_list: list[str | Path],
                      stat_list: list[str | Path],
                      **kwargs):
        spatial_footprint_list = []
        mean_img_list = []
        for ops_file, stat_file in zip(ops_list, stat_list):
            ops = np.load(os.path.abspath(ops_file), allow_pickle=True).item()
            stat = np.load(os.path.abspath(stat_file), allow_pickle=True)
            footprint = extract_suite2p_spatial_footprints(ops, stat)
            mean_img = extract_suite2p_mean_img(ops)
            spatial_footprint_list.append(footprint)
            mean_img_list.append(mean_img)

        return cls(mean_img_list,
                   spatial_footprint_list,
                   **kwargs)


def extract_masknmf_spatial_footprints(dr):
    """
    Given a masknmf demixingresults object, extracts the spatial footprints in a format needed for ROICaT cross-session matching
    """
    a = dr.ac_array.a.cpu().t().coalesce()  # Shape (num_neurons, num_pixels)
    row, col = a.indices()
    vals = a.values()

    row_sum = torch.zeros(a.shape[0], device=a.device)
    row_sum.scatter_reduce_(0, row, vals, reduce="sum")
    per_value_divisors = row_sum[row]
    vals /= per_value_divisors
    vals = torch.nan_to_num(vals, nan=0.0)

    row = row.cpu().numpy()
    col = col.cpu().numpy()
    vals = vals.cpu().numpy()

    shape = a.shape
    curr_csr_scipy = scipy.sparse.coo_matrix((vals, (row, col)), shape=shape).tocsr()
    return curr_csr_scipy


def extract_masknmf_mean_img(dr):
    return dr.pmd_array.mean_img.cpu().numpy()


def extract_suite2p_spatial_footprints(
        ops: np.ndarray,
        stat: np.ndarray,
) -> scipy.sparse.csr_matrix:
    """
    From the suite2p/ROICaT repos
    Returns:
        (scipy.sparse.csr_matrix):
            spatialFootprints (scipy.sparse.csr_matrix):
                Sparse array of shape *(n_roi, frame_height * frame_width)*
                containing the spatial footprints of the ROIs.
    """
    height, width = ops['Ly'], ops['Lx']
    ## Add some code here to infer the height/width of the data from the ops file
    dtype = None
    isInt = np.issubdtype(dtype, np.integer)

    rois_to_stack = []

    for jj, roi in enumerate(stat):
        lam = np.array(roi['lam'], ndmin=1)
        dtype = lam.dtype
        if isInt:
            lam = dtype(lam / lam.sum() * np.iinfo(dtype).max)
        else:
            lam = lam / lam.sum()
        ypix = np.array(roi['ypix'], dtype=np.uint64, ndmin=1)
        xpix = np.array(roi['xpix'], dtype=np.uint64, ndmin=1)

        tmp_roi = scipy.sparse.csr_matrix((lam, (ypix, xpix)), shape=(height, width), dtype=dtype)
        rois_to_stack.append(tmp_roi.reshape(1, -1))

    return scipy.sparse.vstack(rois_to_stack).tocsr()


def extract_suite2p_mean_img(ops) -> np.ndarray:
    mean_img = ops['meanImg']
    return mean_img



