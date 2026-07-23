from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike
from masknmf.utils import Serializer
import torch
from masknmf.utils import torch_select_device
from masknmf.motion_correction.strategies import MotionCorrectionStrategy
from masknmf.motion_correction.rigid_strategy import RigidMotionCorrector
from typing import *
import numpy as np
from masknmf.motion_correction.registration_methods import register_frames_pwrigid
import math
from tqdm import tqdm

class PiecewiseRigidMotionCorrector(MotionCorrectionStrategy, Serializer):
    _serialized = {
        "num_blocks",
        "overlaps",
        "max_rigid_shifts",
        "max_deviation_rigid",
        "template",
        "pixel_weighting",
        "batch_size"
    }

    def __init__(
            self,
            num_blocks: tuple[int, int] = (12, 12),
            overlaps: tuple[int, int] = (5, 5),
            max_rigid_shifts: tuple[int, int] = (15, 15),
            max_deviation_rigid: tuple[int, int] = (2, 2),
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
        self._pixel_weighting = pixel_weighting.astype('float') if pixel_weighting is not None else None

    @property
    def num_blocks(self) -> tuple[int, int]:
        """
        Number of blocks that the image plane is split into, [rows, cols].
        Motion is estimated in each block and then interpolated in 2D space across the entire image plane.
        """
        return self._num_blocks

    @num_blocks.setter
    def num_blocks(self, value):
        value = self._validate_tuple_int_int("num_blocks", value)

        self._template = None
        self._num_blocks = value


    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def pixel_weighting(self) -> None | np.ndarray:
        return self._pixel_weighting

    def _pixel_weighting_tensor(self) -> Optional[torch.Tensor]:
        if self.pixel_weighting is None:
            return None
        return torch.as_tensor(
            self.pixel_weighting, device=self.device, dtype=self.dtype
        )

    @property
    def overlaps(self) -> tuple[int, int]:
        """Number of pixels that overlap between adjacent blocks"""
        return self._overlaps

    @overlaps.setter
    def overlaps(self, value):
        value = self._validate_tuple_int_int("overlaps", value)

        self._template = None
        self._overlaps = value

    @property
    def max_rigid_shifts(self) -> tuple[int, int]:
        """maximum shift for rigid iterations, [rows, cols] before piece-wise rigid registration"""
        return self._max_rigid_shifts

    @max_rigid_shifts.setter
    def max_rigid_shifts(self, value: tuple[int, int]):
        value = self._validate_tuple_int_int("max_rigid_shifts", value)

        self._template = None
        self._max_rigid_shifts = value

    @property
    def max_deviation_rigid(self) -> tuple[int, int]:
        """maximum shift allowed in each piece-wise block relative to the rigid registered frame"""
        return self._max_deviation_rigid

    @max_deviation_rigid.setter
    def max_deviation_rigid(self, value):
        value = self._validate_tuple_int_int("max_deviation_rigid", value)

        self._template = None
        self._max_deviation_rigid = value

    def _correct_singlebatch(
            self,
            reference_frames: np.ndarray | torch.Tensor,
            target_frames: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.template is None:
            raise ValueError(
                "Template is uninitialized"
            )

        if target_frames is not None:
            target_frames = torch.as_tensor(target_frames, device=self.device, dtype=self.dtype)

        reference_frames = torch.as_tensor(reference_frames, device=self.device, dtype=self.dtype)

        template = torch.as_tensor(self.template, device=self.device, dtype=self.dtype)
        if self.pixel_weighting is not None:
            pixel_weighting = torch.as_tensor(self.pixel_weighting, device=self.device, dtype=self.dtype)
        else:
            pixel_weighting = None

        outputs = register_frames_pwrigid(
            reference_frames.to(self.device),
            template,
            self.num_blocks,
            self.overlaps,
            self.max_rigid_shifts,
            self.max_deviation_rigid,
            target_frames=target_frames,
            pixel_weighting=pixel_weighting
        )

        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

    def compute_template(
            self,
            frames: ArrayLike | LazyFrameLoader,
            num_splits_per_iteration: int = 10,
            num_frames_per_split: int = 200,
            num_iterations: int = 1,
    ):
        rigid_strategy = RigidMotionCorrector(
            self.max_rigid_shifts,
            template=self.template if self.template is not None else None,
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


    def _compute_all_shifts(self, reference_movie: ArrayLike) -> np.ndarray:
        """
        Estimate the (n_frames, K1, K2, 2) shift field in a single streaming pass, where K1 and K2 are the number of
        patches in the height and width dimensions of the FOV, respectively.
        """
        ## Maybe add a warning
        if self.template is None:
            self.compute_template(reference_movie)

        num_frames = reference_movie.shape[0]
        num_batches = math.ceil(num_frames / self.batch_size)
        shifts_total = []
        template = torch.as_tensor(self.template, device=self.device, dtype=self.dtype)
        pixel_weighting = self._pixel_weighting_tensor()

        display("extracting all shifts from dataset")
        for k in tqdm(range(num_batches)):
            start = k * self.batch_size
            end = min(start + self.batch_size, num_frames)
            reference_subset = torch.as_tensor(
                reference_movie[start:end], device=self.device, dtype=self.dtype
            )
            if reference_subset.ndim == 2:
                reference_subset = reference_subset[None, ...]

            ## TODO: Make a function that just computes the low-rank shift vectors and does not apply them
            current_shifts = register_frames_pwrigid(reference_subset,
                                                     template,
                                                     self.num_blocks,
                                                     self.overlaps,
                                                     self.max_rigid_shifts,
                                                     self.max_deviation_rigid,
                                                     target_frames=None,
                                                     pixel_weighting=pixel_weighting)[1]
            shifts_total.append(current_shifts.cpu()) ## Move to CPU to avoid overloading

        shifts_total = torch.concatenate(shifts_total, dim=0)
        return shifts_total.cpu()

    def motion_correct(
            self,
            reference_movie: ArrayLike,
            target_movie: ArrayLike | None = None,
    ) -> ArrayLike:
        ## TODO: Update this with actual implementation
        shift_data = self._compute_all_shifts(reference_movie)
        return shift_data




