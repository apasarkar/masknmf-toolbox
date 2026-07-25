import math

from masknmf.utils import Serializer
from masknmf.utils import torch_select_device
import torch
from masknmf.motion_correction.strategies import MotionCorrectionStrategy
from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, LazyFrameLoader
from tqdm import tqdm
from masknmf.motion_correction.registration_arrays import OphysArray
import copy

class GradientMotionCorrector(MotionCorrectionStrategy, Serializer):
    """
    This is a motion corrector designed to correct small (<2 pixel) jitter. Primarily for culture imaging data
    """
    _serialized = {
        "template",
        "batch_size"
    }
    def __init__(
            self,
            template: np.ndarray | torch.Tensor | None = None,
            batch_size: int = 200,
            pixel_weighting: np.ndarray | torch.Tensor | None = None,
            device: str = "auto",
    ):
        self._device = torch_select_device(device)
        self.batch_size = batch_size
        self._pixel_weighting = None
        self._template = None

        self.pixel_weighting = pixel_weighting
        self.template = template


    @property
    def dtype(self):
        """
        What dtype is used for computations
        """
        return torch.float32

    @property
    def batch_size(self) -> int:
        """get or set the batch size, the number of frames sent to the GPU in batches for motion correction"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if not isinstance(value, int):
            raise ValueError(f"`batch_size` must be an <int>, you passed: {value}")
        self._batch_size = value

    @property
    def template(self) -> None | torch.Tensor:
        return self._template

    @template.setter
    def template(self, new_template: None | np.ndarray | torch.Tensor):
        if new_template is None:
            self._template = new_template
        elif isinstance(new_template, (np.ndarray, torch.Tensor)):
            self._template = torch.as_tensor(new_template, dtype=self.dtype, device=self.device)
            self._compute_gradient_matrix()
        else:
            raise TypeError(f"template should be None, np.ndarray, or torch.Tensor")

    @property
    def pixel_weighting(self) -> None | torch.Tensor:
        return self._pixel_weighting

    @pixel_weighting.setter
    def pixel_weighting(self, weighting: np.ndarray | torch.Tensor):
        if weighting is None:
            self._pixel_weighting = None
        elif isinstance(weighting, (np.ndarray, torch.Tensor)):
            self._pixel_weighting = torch.as_tensor(weighting, dtype=self.dtype, device=self.device)
        else:
            raise TypeError(f"pixel_weighting should be None, np.ndarray, or torch.Tensor")

    def correction_routine(
            self,
            reference_frames: np.ndarray | torch.Tensor | None,
            target_frames: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Basic model: we want to register a frame F_1 to a template frame, F_0:

        F_0(x, y) + grad_F0 @ w = alpha * F_1(x, y)
        where w is a 2D vector of gradient coefficients indicating the size of the shift w.r.t. the template F0.
        alpha is a scalar (to account for scale differences between F_0 and F_1 because of things like photobleaching)

        This models assumes that the shift between F_0(x, y) and alpha * F_1(x, y) can be explained by the gradient of F0.

        By projecting both sides of this equation to the subspace orthogonal to F_0, we can solve for the vector
        w / alpha:

        w/alpha = projection of Orth_F0(F1) onto Orth_F0(grad_F0)

        We can then estimate alpha as well as the least squares coefficient relating F_0 and F_1. "w" describes the gradient step
        in each dimension at the scale of the template image.

        Note that the actual correction applied to frame F_1(x, y) is:

        F_1(x, y) - gradF_0 @ (w / alpha)

        Where this approach breaks down:
        - Lots of noise relative to the prominent features of the template/reference image
        - Large dF/F

        Args:
            reference_frames (np.ndarray): Shape (num_frames, height, width). Frames used to decide gradient shifts in each dimension
            target_frames (np.ndarray): Shape (num_frames, height, width). Frames to which the correction is applied
                If None, motion correction is applied to reference_frames
        Returns:
            Corrected frame (torch.Tensor): Motion corrected frame. Shape (num_frames, height, width)
            Shift (torch.Tensor): The shift vectors in gradient space, at the scale of the template image; this is the "w" vector
            alpha (torch.Tensor): The scale mismatch between the reference frame and template. w / alpha gives the actual gradient coefficients
                applied to do the correction, as mentioned above.
        """

        if self.template is None:
            raise ValueError(
                "Template is uninitialized"
            )

        if target_frames is not None:
            target_frames = torch.as_tensor(target_frames, device=self.device, dtype=self.dtype)

        reference_frames = torch.as_tensor(reference_frames, device=self.device, dtype=self.dtype)
        num_frames, height, width = reference_frames.shape
        gradient = torch.as_tensor(self._gradient, device=self.device, dtype=self.dtype)
        gradient_projector = torch.as_tensor(self._gradient_projector, device=self.device, dtype=self.dtype)

        ref_flat = reference_frames.permute(1, 2, 0).reshape(-1, num_frames)  # [H*W, F]
        if target_frames is not None:
            target_flat = target_frames.permute(1, 2, 0).reshape(-1, num_frames)
        else:
            target_flat = None
        gradient_step = gradient_projector @ ref_flat  # [2, F]
        net_correction = gradient @ gradient_step

        if target_frames is not None:
            result = target_flat - net_correction
        else:
            result = ref_flat - net_correction

        """
        The actual "gradient step" from the observed data --> template requires more care: need to 
        make sure the mean image matches the scale of the image in question (photobleaching etc. can affect this). 
        If they don't match, the gradients will be "off" by some multiplicative factor. 

        So we do a least squares regression of the observed data onto the mean image
        """
        template = self.template
        if self.pixel_weighting is not None:
            template = template * self.pixel_weighting
        alpha = torch.sum(reference_frames * template[None, ...], dim=(1, 2)) / torch.sum(reference_frames ** 2,
                                                                                          dim=(1, 2))
        shift = gradient_step * alpha[None, :]

        shift = torch.nan_to_num(shift, nan=0.0, posinf=0.0, neginf=0.0).permute(1, 0) #Shape (frames, 2)

        result = result.reshape(height, width, num_frames).permute(2, 0, 1)

        return result, shift, alpha

    @property
    def gradient(self) -> torch.Tensor | None:
        """
        A (pixels, 2) tensor describing the gradients in X and Y dimensions
        """
        return self._gradient

    @property
    def gradient_projector(self) -> torch.Tensor | None:
        return self._gradient_projector

    def compute_gradient_step(self,
                              shifts: torch.Tensor,
                              alpha: torch.Tensor):
        """
        The shifts computed by the gradient strategy are at the scale of the template image. This routine computes the
        gradient coefficients that are used to correct the i-th frame of data, this routine returns w / alpha

        Args:
            shifts (torch.Tensor): Shape (num_frames, 2). The gradient shifts in each dimension, at the scale of the template
            alpha (torch.Tensor): Shape (num_frames). The scale mismatch between the template and the frames
        """
        gradient_steps = shifts / alpha[:, None]
        gradient_steps = torch.nan_to_num(gradient_steps, nan=0.0, posinf=0.0, neginf=0.0)
        return gradient_steps

    def _correct_singlebatch(
            self,
            reference_frames: np.ndarray,
            target_frames: Optional[np.ndarray],
            apply_alpha_correction: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        result, shifts, _ = self.correction_routine(reference_frames,
                                                         target_frames)


        return result.cpu().numpy(), shifts.cpu().numpy()

    def compute_template(
            self,
            frames: ArrayLike
    ):
        self.template = frames[0].squeeze(0)

    def _compute_gradient_matrix(self):
        """
        Computes the gradient design matrix used for the motion correction
        :return:
        """
        f0 = self.template
        dfdx_orig = torch.nn.functional.pad(
            (f0[:, 2:] - f0[:, :-2]) / 2, (1, 1), mode="constant"
        )  # x-gradient [H, W]
        dfdy_orig = torch.nn.functional.pad(
            (f0[2:, :] - f0[:-2, :]) / 2, (0, 0, 1, 1), mode="constant"
        )  # y-gradient [H, W]



        if self.pixel_weighting is not None:
            f0 = f0 * self.pixel_weighting
            mask =self.pixel_weighting
            dfdx = dfdx_orig * mask
            dfdy = dfdy_orig * mask
        else:
            dfdx = dfdx_orig
            dfdy = dfdy_orig

        f0_flat = f0.flatten()  # [H*W]
        f0_norm_sq = f0_flat @ f0_flat

        def orthogonalize_against_template(v):
            """
            This routine orthogonalizes against the mean image. This is important because it
            makes the shift estimation insensitive to
            """
            return v - (f0_flat @ v) / f0_norm_sq * f0_flat

        dfdx_flat = orthogonalize_against_template(dfdx.flatten())
        dfdy_flat = orthogonalize_against_template(dfdy.flatten())

        A = torch.stack([dfdx_flat, dfdy_flat], dim=1).float()  # [H*W, 2]
        AtA_inv = torch.linalg.inv(A.T @ A)  # [2, 2]
        A_projector = AtA_inv @ A.T  # [2, H*W]

        self._gradient = torch.stack([dfdx_orig.flatten(), dfdy_orig.flatten()], dim=1).to(self.dtype).to(self.device) ## The design matrix doesn't change
        self._gradient_projector = A_projector.to(self.dtype).to(self.device) ## The projector can be orthogonalized w.r.t. f0

    def _extract_all_shifts(self,
                            reference_movie: OphysArray):
        num_frames, height, width = reference_movie.shape
        batch_size = self.batch_size
        num_iters = math.ceil(num_frames / batch_size)

        gradient_steps = []
        total_shifts = []
        for k in tqdm(range(num_iters)):
            start = k * batch_size
            end = min(start + batch_size, num_frames)
            subset = torch.as_tensor(reference_movie._get(slice(start, end), include_mean=True), device=self.device,
                                     dtype=self.dtype)
            frames, shifts, scale = self.correction_routine(subset)
            gradient_step = self.compute_gradient_step(shifts, scale)
            gradient_steps.append(gradient_step.cpu())
            total_shifts.append(shifts.cpu())

        gradient_steps = torch.concatenate(gradient_steps, dim=0)  # Shape (frames, 2)
        total_shifts = torch.concatenate(total_shifts, dim=0)
        return gradient_steps, total_shifts

    def motion_correct(self,
                       reference_movie: OphysArray) -> ArrayLike:

        gradient_steps, shifts = self._extract_all_shifts(reference_movie)
        return GradientRegistrationArray(reference_movie,
                                         copy.deepcopy(self),
                                         gradient_steps,
                                         shifts)



class GradientRegistrationArray(ArrayLike):
    """
    This is a special registration array which provides fast slicing in all dimensions (pixels AND frames)
    The key idea is that the per-frame correction (which uses the gradient of the template image) is low-rank,
    so we can adaptively compute frames/pixels of this movie very quickly if we just store the low-rank info

    """
    def __init__(self,
                 raw_data: OphysArray,
                 strategy: GradientMotionCorrector,
                 gradient_steps: torch.Tensor,
                 shifts: torch.Tensor,
                 output_device: str = "cpu"):

        self._raw_data = raw_data
        self._strategy = strategy
        self._output_device = output_device
        self._gradient_steps_mean = None
        self._gradient = self.strategy.gradient.reshape(self.shape[1], self.shape[2], 2).to(self.strategy.device).to(self.dtype)
        self._shifts = shifts
        self._gradient_steps = gradient_steps.to(self.strategy.device).to(self.dtype)
        self._gradient_steps_mean = torch.mean(self._gradient_steps, dim=0)
        self._include_mean = True

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.raw_data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.strategy.dtype

    @property
    def output_device(self) -> str:
        return self._output_device

    @output_device.setter
    def output_device(self, new_device:str):
        self._output_device = new_device

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def raw_data(self) -> ArrayLike:
        return self._raw_data

    @property
    def include_mean(self) -> bool:
        return self._include_mean

    @include_mean.setter
    def include_mean(self, new_flag: bool):
        self._include_mean = new_flag

    @property
    def strategy(self) -> GradientMotionCorrector:
        return self._strategy

    @property
    def shifts(self) -> torch.Tensor:
        """
        The shift estimates for all frames of the data. THe scale of these shifts are relative to the template used; see
        GradientMotionCorrector for details
        Shape (num_frames, 2)
        """
        return self._shifts

    @property
    def gradient_steps(self) -> torch.Tensor | None:
        """
        The gradient steps (in the height and width dimensions respectively) that are applied to each frame of the
        input data shape (frames, 2)
        """
        return self._gradient_steps


    @property
    def gradient(self) -> torch.Tensor | None:
        """
        The template gradient image. The first image (gradient[:, :, 0]) is the height gradient and the second image is the width gradient
        Shape (height, width, 2)
        """
        return self._gradient

    def __getitem__(self,
                    item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]) -> torch.Tensor:

        data_subset = torch.as_tensor(self.raw_data._get(item, include_mean=self.include_mean), device=self.output_device, dtype=self.dtype)
        frame_indexer, item = self._parse_indices(item)

        gradient_step_used = self.gradient_steps[frame_indexer, :]
        if gradient_step_used.ndim == 1:
            gradient_step_used = gradient_step_used[None, :]

        if not self.include_mean:
            if gradient_step_used.ndim > self._gradient_steps_mean.ndim:
                grad_sub = self._gradient_steps_mean[None, :]
            else:
                grad_sub = self._gradient_steps_mean
            gradient_step_used -= grad_sub

        # Check if spatial cropping occurred, deal with factorized tensors appropriately
        if isinstance(item, tuple):
            gradient_crop = self.gradient[item[1:]]
        else:
            gradient_crop = self.gradient

        grad_movie = gradient_crop @ gradient_step_used.T
        grad_movie = grad_movie.movedim(-1, 0).to(self.output_device)

        return data_subset - grad_movie


