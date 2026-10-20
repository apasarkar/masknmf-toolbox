from masknmf import ArrayLike


class OphysArray(ArrayLike):

    def __init__(self,
                 dataset: ArrayLike,
                 negative_indicator: bool = True,
                 include_mean: bool = True,
                 device='cuda',
                 batch_size: int = 200):
        """
        Array-like object for viewing inverted, mean subtracted, and/or raw optical physiology data
        Args:
            dataset (masknmf.ArrayLike): Shape (num_frames, height, width)
            negative_indicator (bool): True if indicator is negatively tuned, else False
            include_mean (bool): If True, includes the mean into the "getitem" call. If false, getitem shows the "mean subtracted" movie
            device (str): Which device to perform computations/return the tensor from getitem
        """
        self._dataset = dataset
        self._device = device
        self._batch_size = batch_size
        self._negative_indicator = negative_indicator
        self._compute_mean()
        self._include_mean = include_mean

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: int):
        self._batch_size = new_size

    def _compute_mean(self):
        num_frames = self.shape[0]
        cumulated_mean = torch.zeros(self.shape[1], self.shape[2], dtype=self.dtype, device=self.device)
        num_batches = math.ceil(self.shape[0] / self.batch_size)
        for ind in range(num_batches):
            start_pt = ind * self.batch_size
            end_pt = min(self.shape[0], start_pt + self.batch_size)
            data_subset = self._dataset[start_pt:end_pt]
            if isinstance(data_subset, np.ndarray):
                subset = torch.from_numpy(data_subset).to(self.device).to(self.dtype)
            elif isinstance(data_subset, torch.Tensor):
                subset = data_subset.to(self.device).to(self.dtype)
            else:
                raise ValueError("Calling getitem on dataset should return either a torch tensor or np.ndarray")
            cumulated_mean += torch.sum(subset, dim=0) / num_frames
        self._mean_image = cumulated_mean

    @property
    def include_mean(self) -> bool:
        return self._include_mean

    @include_mean.setter
    def include_mean(self, new_flag: bool):
        self._include_mean = new_flag

    @property
    def negative_indicator(self) -> bool:
        return self._negative_indicator

    @property
    def device(self) -> str:
        return self._device

    @property
    def mean_image(self) -> torch.Tensor:
        return self._mean_image

    @property
    def dtype(self) -> torch.dtype:
        """
        data type
        """
        return torch.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._dataset.shape

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def __getitem__(self,
                    item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range]) -> torch.Tensor:
        return self._get(item, include_mean=self.include_mean)

    def _get(self,
             item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
             include_mean: bool | None = None) -> torch.Tensor:
        ## Private helper method designed to avoid race conditions associated with state variables like include_mean
        if include_mean is None:
            include_mean = self.include_mean

        frame_indexer, item = self._parse_indices(item)
        data_subset = torch.as_tensor(self._dataset[item]).to(self.device).to(self.dtype) ## Parse indices first to ensure consistent output

        # Check if spatial cropping occurred, deal with mean image accordingly
        if isinstance(item, tuple):
            mean_crop = self._mean_image[item[1:]]
        else:
            mean_crop = self._mean_image

        if data_subset.ndim > mean_crop.ndim: #This means that there is a temporal dimension, which means we need to broadcast
            mean_crop = mean_crop[None, ...]

        if self.negative_indicator:
            data_subset *= -1
            if include_mean:
                data_subset += 2 * mean_crop
            else:
                data_subset += mean_crop
        else:
            if include_mean:
                pass
            else:
                data_subset -= mean_crop
        return data_subset
