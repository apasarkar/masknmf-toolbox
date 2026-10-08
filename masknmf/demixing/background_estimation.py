import torch


class RingModel:
    def __init__(
            self,
            fov_height: int,
            fov_width: int,
            radius: int,
            device: torch.device | str = "cpu",
    ):
        """
        Ring Model object manages the state of the ring model during the model fit phase.

        Args:
            fov_height (int): The height dimension of the imaging field of view
            fov_width (int): The width dimension of the imaging field of view
            radius (int): the ring radius
            device (str): which device the pytorch data lies on
        """
        self._shape = (fov_height, fov_width)
        self._radius = radius
        self._device = device
        self._kernel = self._construct_ring_kernel()
        self.weights = torch.ones((fov_height * fov_width), device=device)
        self.support = torch.ones(
            (self.shape[0] * self.shape[1]), device=self.device, dtype=torch.float32
        )

    def _construct_ring_kernel(self) -> torch.Tensor:
        # Create a grid of coordinates (y, x) relative to the center
        range_values = torch.arange(
            2 * self.radius + 1, device=self.device
        )  # Guarantees kernel on right device
        y, x = torch.meshgrid(range_values, range_values, indexing="ij")
        y = y - self.radius
        x = x - self.radius

        # Calculate the distance from the center (radius, radius)
        dist = torch.sqrt(x.float() ** 2 + y.float() ** 2)

        # Create the ring kernel: 1 if the distance is exactly `radius`, otherwise 0
        ring_kernel = (dist >= self.radius - 0.5) & (dist <= self.radius + 0.5)
        return ring_kernel.float()

    @property
    def kernel(self) -> torch.Tensor:
        return self._kernel

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self._device

    @property
    def radius(self):
        return self._radius

    @property
    def weights(self):
        """
        The ring model uses a constant weight assumption: that pixel "i" of the data can be explained as a scaled
        average of the pixels in a ring surrounding pixel "i". This is enforced by a diagonal weight matrix: d_weight.

        Returns:
            d_weights (torch.sparse_coo_tensor): (fov_height*fov_width, fov_height*fov_width) diagonal matrix
        """
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        """
        Sets the weights
        Args:
            new_weights (torch.Tensor): Shape (fov_height*fov_width)
        """
        self._weights = new_weights.clone().to(self.device)

    @property
    def support(self):
        """
        The ring model only operates on pixels that do not contain spatial footprints. This is enforced by a diagonal
        mask matrix, D_{mask}. The i-th entry  is 0 if pixel i contains neural footprints, otherwise it is 1

        Returns:
            d_mask (torch.sparse_coo_tensor): (fov_height*fov_width, fov_height*fov_width) diagonal matrix represe
        """
        return self._support

    @support.setter
    def support(self, new_mask):
        """
        Args:
            new_mask (torch.Tensor): Shape (fov_height*fov_width), index i is 0 if pixel i contains neural signal, otherwise it is 0
        """
        self._support = new_mask.clone().to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies the ring model to a stack of 2D images using FFT-based convolution.

        Args:
            images (torch.Tensor): Shape (pixels, num_frames)

        Returns:
            ring_outputs (torch.Tensor): Shape (pixels, num_frames)
        """
        images_masked = self.support[:, None] * images  # (pixels, num_frames)

        # Reshape to (fov_height, fov_width, frames)
        images_masked_3_d = torch.reshape(images_masked, (self.shape[0], self.shape[1], -1))

        # Shape: (frames, H, W)
        images_masked_3_d = torch.permute(images_masked_3_d, (2, 0, 1))

        # Add channel dimension: (frames, 1, H, W)
        images_masked_3_d = images_masked_3_d.unsqueeze(1)

        # Pad input so that convolution result is defined for all pixels (esp. near edges)
        pad = self.radius
        images_padded = torch.nn.functional.pad(images_masked_3_d, (pad, pad, pad, pad))  # (frames, 1, H+2r, W+2r)

        frames, _, H_pad, W_pad = images_padded.shape

        # Pad kernel to match padded image size
        kh, kw = self.kernel.shape
        kernel_padded = torch.zeros((frames, 1, H_pad, W_pad), dtype=torch.float32, device=images.device)
        # Place kernel at top-left corner
        kernel_padded[:, 0, :kh, :kw] = self.kernel
        # Center the kernel (equivalent to fftshift)
        kernel_padded = torch.roll(kernel_padded, shifts=(-kh // 2, -kw // 2), dims=(-2, -1))

        # FFT-based convolution
        images_fft = torch.fft.rfft2(images_padded.float(), dim=(-2, -1))
        kernel_fft = torch.fft.rfft2(kernel_padded.float(), dim=(-2, -1))
        fft_product = images_fft * kernel_fft
        convolved = torch.fft.irfft2(fft_product, s=(H_pad, W_pad), dim=(-2, -1))  # (frames, 1, H+2r, W+2r)

        # Crop back to original image size
        convolved = convolved[:, :, pad:-pad, pad:-pad]  # (frames, 1, H, W)

        # Remove channel dimension
        convolved = convolved.squeeze(1)  # (frames, H, W)

        # Convert back to (pixels, frames)
        convolved = torch.permute(convolved, (1, 2, 0))  # (fov_height, fov_width, frames)

        convolved = convolved.reshape((self.shape[0] * self.shape[1], -1))

        # Apply output weights
        return self.weights[:, None] * convolved
    