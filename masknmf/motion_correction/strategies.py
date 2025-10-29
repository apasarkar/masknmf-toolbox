import math
import random
import warnings

import torch
from typing import *

import numpy as np
from tqdm import tqdm

import masknmf
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.utils import torch_select_device
from .registration_methods import register_frames_rigid, register_frames_pwrigid

class MotionCorrectionStrategy:
    """base class for motion correction strategies."""

    def __init__(
            self,
            template: Optional[np.ndarray] = None,
            batch_size: int = 200,
            device: str = "auto",
    ):
        self._device = torch_select_device(device)
        self._template = torch.from_numpy(template).float() if template is not None else None
        self._batch_size = batch_size

    @property
    def template(self) -> Union[None, torch.tensor]:
        return self._template

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def pixel_weighting(self) -> Union[None, torch.Tensor]:
        raise NotImplementedError

    def _correct_singlebatch(
            self,
            reference_frames: np.ndarray,
            target_frames: Optional[np.ndarray] | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function should contain the core logic for motion correcting "n" frames, without any batching logic needed.

        All arrays (movie frames) are sent to the device within this call, and must be garbage collected by the end
        of this call. The idea is that a big full movie will never fit on the GPU, so `_correct_batch` manages sending
        a batch of frames to be corrected along with any other required arrays (such as the template) to the device and
        garbage collecting afterward.

        Args:
            reference_frames (np.ndarray): Shape (num_frames, height, width).
            target_frames (np.ndarray): Shape (num_frames, height, width).
        Returns:
            - motion corrected frames (np.ndarray). Shape (num_frames, height, width)
            - shifts (np.ndarray). Shape depends on what information needs to be conveyed here. Dimension 0 should
                still be number of frames though.
        """
        raise NotImplementedError

    def correct(
        self,
        reference_movie_frames: np.ndarray,
        target_movie_frames: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Apply motion correction. Reference frames is the set of frames that we align to the template (to learn
        shifts, motion displacement fields, etc.) and target_frames is the image stack to which we apply the
        motion stabilization transformation. If target_frames is unspecified, the motion correction is applied to
        reference_frames.

        Two returned values: (1) the motion corrected data (2) the shift (or displacement field information).
        The data format for (2) varies based on method used
        """

        if reference_movie_frames.ndim == 2:
            reference_movie_frames = reference_movie_frames[None, ...]
            if target_movie_frames is not None:
                target_movie_frames = target_movie_frames[None, ...]

        num_iters = math.ceil(reference_movie_frames.shape[0] / self.batch_size)
        registered_frame_outputs = []
        frame_shift_outputs = []

        for k in range(num_iters):
            start = k * self.batch_size
            end = min(start + self.batch_size, reference_movie_frames.shape[0])
            reference_subset = reference_movie_frames[start:end]
            if target_movie_frames is not None:
                target_subset = target_movie_frames[start:end]
            else:
                target_subset = None

            if reference_subset.ndim == 2:
                reference_subset = reference_subset[None, :, :]
            if target_subset is not None and target_subset.ndim == 2:
                target_subset = target_subset[None, :, :]

            reg_output = self._correct_singlebatch(
                reference_frames=reference_subset,
                target_frames=target_subset,
            )

            registered_frame_outputs.append(reg_output[0])
            frame_shift_outputs.append(reg_output[1])

        moco_output = np.concatenate(registered_frame_outputs, axis=0)
        shift_output = np.concatenate(frame_shift_outputs, axis=0)
        return moco_output, shift_output

    def compute_template(
            self,
            frames: Union[masknmf.ArrayLike, masknmf.LazyFrameLoader],
            num_splits_per_iteration: int = 10,
            num_frames_per_split: int = 200,
            num_iterations: int = 3,
        ):

        if num_iterations <= 0:
            raise ValueError(f"`num_iterations` must be >= 1`, you passed: {num_iterations}")


        # Step 1: Initial Template (Mean Image)
        if self.template is None:
            # template not specified by user, estimate using just the first 500 frames of the movie
            if frames.shape[0] < 500:
                warnings.warn("Using less than 500 frames to create registration template")

            # account for use cases with very few frames, ex: spatial transcriptomics
            num_frames_template = min(frames.shape[0], 500)

            frames_loaded = frames[:num_frames_template]
            template = torch.from_numpy(np.median(frames_loaded, axis=0))
            self._template = template

        ## Prepare the template estimation pipeline by establishing the chunks of data to sample
        slice_list = self._compute_frame_chunks(frames.shape[0], num_frames_per_split)
        num_splits_to_sample = min(num_splits_per_iteration, len(slice_list))

        for pass_iter_rigid in range(num_iterations):
            slices_sampled = random.sample(slice_list, num_splits_to_sample)
            template_list = []
            for j in tqdm(range(num_splits_to_sample)):
                corrected_frames = self.correct(frames[slices_sampled[j]])[0]
                template = np.mean(corrected_frames, axis=0)
                template_list.append(template)

            self._template = torch.from_numpy(
                np.median(
                    np.stack(template_list, axis=0), axis=0
                )
            )
        torch.cuda.empty_cache()

    def _compute_frame_chunks(self, num_frames: int, frames_per_split: int) -> list:
        """
        Sets a partition of the frames into individual contiguous chunks of frames.

        Args:
            num_frames (int): The number of frames in the dataset
            frames_per_split (int): The number of frames in each chunk of data which we use to estimate a local template
        Returns:
            slice_list (list): A list of slices, each describing a start and end for a given chunk of data which can be used to
                refine the template estimate.
        """

        start_pts = list(range(0, num_frames, frames_per_split))
        if start_pts[-1] > num_frames - frames_per_split and start_pts[-1] > 0:
            start_pts[-1] = num_frames - frames_per_split

        slice_list = [
            slice(start_pts[i], min(num_frames, start_pts[i] + frames_per_split))
            for i in range(len(start_pts))
        ]
        return slice_list


class RigidMotionCorrector(MotionCorrectionStrategy):
    def __init__(
        self,
        max_shifts: Tuple[int, int],
        template: Optional[np.ndarray] = None,
        pixel_weighting: Optional[np.ndarray] = None,
        batch_size: int = 200,
        device: str = "auto",
    ):
        super().__init__(template, batch_size=batch_size, device=device)

        self._max_shifts = max_shifts
        self._pixel_weighting = torch.from_numpy(pixel_weighting).float() if pixel_weighting is not None else None

    @property
    def max_shifts(self) -> Tuple[int, int]:
        return self._max_shifts

    @max_shifts.setter
    def max_shifts(self, value: Tuple[int, int]):
        value = tuple(map(int, value))
        if len(value) != 2:
            raise ValueError(
                f"`max_shifts` must be a tuple of int, i.e. (int, int), of size 2. You have passed: {value}"
            )

        self._max_shifts = value

    @property
    def pixel_weighting(self) -> Union[None, torch.tensor]:
        return self._pixel_weighting

    def _correct_singlebatch(
            self,
            reference_frames: np.ndarray,
            target_frames: Optional[np.ndarray] | None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.template is None:
            raise ValueError(
                "Template is uninitialized"
            )

        if target_frames is not None:
            target_frames = torch.from_numpy(target_frames).to(self.device).float()

        reference_frames = torch.from_numpy(reference_frames).to(self.device).float()

        outputs = register_frames_rigid(
            reference_frames.to(self.device).float(),
            self.template.to(self.device).float(),
            self.max_shifts,
            target_frames=target_frames,
            pixel_weighting=self.pixel_weighting.to(self.device).float() if self.pixel_weighting is not None else None,
        )
        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

class PiecewiseRigidMotionCorrector(MotionCorrectionStrategy):
    def __init__(
        self,
        num_blocks: Tuple[int, int],
        overlaps: Tuple[int, int],
        max_rigid_shifts: Tuple[int, int],
        max_deviation_rigid: Tuple[int, int],
        template: Optional[np.ndarray] = None,
        pixel_weighting: Optional[np.ndarray] = None,
        batch_size: int = 200,
        device: str = "auto",
    ):
        super().__init__(template, batch_size=batch_size, device=device)
        self._num_blocks = num_blocks
        self._overlaps = overlaps
        self._max_rigid_shifts = max_rigid_shifts
        self._max_deviation_rigid = max_deviation_rigid
        self._pixel_weighting = torch.from_numpy(pixel_weighting).float() if pixel_weighting is not None else None

    @property
    def num_blocks(self) -> Tuple[int, int]:
        return self._num_blocks

    @property
    def pixel_weighting(self) -> Optional[torch.tensor]:
        return self._pixel_weighting

    @property
    def overlaps(self) -> Tuple[int, int]:
        return self._overlaps

    @property
    def max_rigid_shifts(self) -> Tuple[int, int]:
        return self._max_rigid_shifts

    @property
    def max_deviation_rigid(self) -> Tuple[int, int]:
        return self._max_deviation_rigid

    def _correct_singlebatch(
            self,
            reference_frames: np.ndarray,
            target_frames: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.template is None:
            raise ValueError(
                "Template is uninitialized"
            )

        if target_frames is not None:
            target_frames = torch.from_numpy(target_frames).to(self.device).float()

        reference_frames = torch.from_numpy(reference_frames).to(self.device).float()

        outputs = register_frames_pwrigid(
            reference_frames.to(self.device),
            self.template.to(self.device),
            self.num_blocks,
            self.overlaps,
            self.max_rigid_shifts,
            self.max_deviation_rigid,
            target_frames=target_frames,
            pixel_weighting=self.pixel_weighting.to(self.device) if self.pixel_weighting is not None else None
        )

        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

    def compute_template(
            self,
            frames: Union[masknmf.ArrayLike, masknmf.LazyFrameLoader],
            num_splits_per_iteration: int = 10,
            num_frames_per_split: int = 200,
            num_iterations:int = 1,
    ):
        rigid_strategy = RigidMotionCorrector(
            self.max_rigid_shifts,
            template=self.template.cpu().numpy() if self.template is not None else None,
            pixel_weighting=self.pixel_weighting,
            batch_size=self.batch_size,
            device=self.device,
        )

        rigid_strategy.compute_template(
            frames,
        )

        self._template = rigid_strategy.template

        super().compute_template(
            frames,
            num_splits_per_iteration=num_splits_per_iteration,
            num_frames_per_split=num_frames_per_split,
            num_iterations=num_iterations,
        )
        torch.cuda.empty_cache()
