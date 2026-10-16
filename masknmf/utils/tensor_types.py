from typing import Annotated, TypeAlias

import torch

SparseCOOTensor: TypeAlias = Annotated[torch.Tensor, torch.sparse_coo]
SparseCSRTensor: TypeAlias = Annotated[torch.Tensor, torch.sparse_csr]