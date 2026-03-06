import torch
import random
import numpy as np
from torch.utils.data import Dataset


def set_seed(seed: int = 0) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep default deterministic behavior lightweight for toy experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sinusoidal_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10_000
) -> torch.Tensor:
    """
    Compute sinusoidal timestep embeddings as commonly used in diffusion models.

    This function maps a batch of scalar timesteps to a higher-dimensional
    embedding space using fixed sinusoidal features. The embedding consists
    of sine and cosine functions at exponentially spaced frequencies,
    similar to positional encodings used in Transformers and DDPM models.

    Args:
        t (torch.Tensor):
            A 1D tensor of shape (B,) containing timesteps.
            Typically integer (dtype=torch.long), but may also be float.
            Each value represents a diffusion timestep.

        dim (int):
            Dimension of the output embedding. Must be >= 2.

        max_period (int, optional):
            Controls the minimum frequency of the embeddings.
            Larger values produce lower minimum frequencies.
            Default is 10_000.

    Returns:
        torch.Tensor:
            A tensor of shape (B, dim) containing the sinusoidal embeddings
            corresponding to each timestep.

    Raises:
        ValueError:
            If `t` is not a 1D tensor of shape (B,).
        ValueError:
            If `dim` is less than 2.

    Notes:
        - The embedding is constructed using sine and cosine functions
          with geometrically spaced frequencies.
        - If `dim` is odd, the final dimension may be zero-padded.
        - The operation is deterministic and contains no learnable parameters.
    """
    if t.ndim != 1:
        raise ValueError(f"`t` must be 1D with shape (B,), got shape={tuple(t.shape)}")
    if dim < 2:
        raise ValueError(f"`dim` must be >= 2, got dim={dim}")

    half = dim // 2
    t = t.float()
    device = t.device

    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=device))
        * torch.arange(half, dtype=torch.float32, device=device)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device)], dim=1)
    return emb


class GaussianDataset(Dataset):
    """
    Dataset representing a Gaussian distribution or a mixture of Gaussians.

    This dataset generates synthetic samples drawn from either:
        - A single Gaussian distribution
        - A mixture of multiple Gaussian components with uniform weights

    It supports:
        - Arbitrary number of mixture components
        - Configurable means and standard deviations
        - Shared standard deviation across components
        - Preset configurations for common 1D and 2D cases

    Preset Behavior (if preset=True):
        - dim=1:
            * 3 components
            * Means: [[1.0], [2.0], [3.0]]
            * Shared std: 0.05

        - dim=2:
            * 25 components arranged on a 5×5 equally spaced grid
            * Grid bounds controlled by grid_min_2d and grid_max_2d
            * Shared std controlled by std_2d

    Each sample consists of:
        - x: data point drawn from selected Gaussian component
        - y: integer component index

    Attributes:
        n (int):
            Total number of samples.

        dim (int):
            Dimensionality of each data point.

        n_components (int):
            Number of Gaussian mixture components.

        means (torch.Tensor):
            Tensor of shape (n_components, dim) containing component means.

        stds (torch.Tensor):
            Tensor of shape (n_components, dim) containing component standard deviations.

    """

    def __init__(
        self,
        n: int,
        dim: int,
        n_components: int | None = None,
        means=None,
        stds=None,
        shared_std: bool = True,
        preset: bool = True,
        grid_size_2d: int = 5,
        grid_min_2d: float = -2.0,
        grid_max_2d: float = 2.0,
        std_2d: float = 0.2,
    ) -> None:
        """
        Initialize the GaussianDataset.

        Args:
            n (int):
                Total number of samples to generate.

            dim (int):
                Dimensionality of each sample.

            n_components (int | None, optional):
                Number of Gaussian mixture components.
                Required if preset=False and means are not provided.

            means (array-like or torch.Tensor, optional):
                Mean vectors for each component.
                Expected shape: (n_components, dim).

            stds (float, int, array-like, or torch.Tensor, optional):
                Standard deviations for each component.
                If shared_std=True, must be a scalar.
                Otherwise, must match shape (n_components, dim).

            shared_std (bool, optional):
                If True, a single scalar std is shared across all components.
                If False, each component may have its own std.
                Default is True.

            preset (bool, optional):
                If True, automatically constructs predefined mixtures for dim=1 or dim=2.
                Default is True.

            grid_size_2d (int, optional):
                Number of grid points per axis for 2D preset.
                Default is 5.

            grid_min_2d (float, optional):
                Minimum coordinate value for 2D grid preset.
                Default is -2.0.

            grid_max_2d (float, optional):
                Maximum coordinate value for 2D grid preset.
                Default is 2.0.

            std_2d (float, optional):
                Shared standard deviation used in 2D preset.
                Default is 0.2.

        """
        self.n = int(n)
        self.dim = int(dim)
        if self.n <= 0:
            raise ValueError(f"`n` must be > 0, got {self.n}")
        if self.dim not in (1, 2):
            raise ValueError(f"Only dim in {{1,2}} is supported, got {self.dim}")

        if preset:
            if self.dim == 1:
                self.means = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
                self.n_components = 3
                self.stds = torch.full((self.n_components, self.dim), 0.05, dtype=torch.float32)
            else:
                axis = torch.linspace(grid_min_2d, grid_max_2d, grid_size_2d, dtype=torch.float32)
                grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
                self.means = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
                self.n_components = self.means.shape[0]
                self.stds = torch.full((self.n_components, self.dim), float(std_2d), dtype=torch.float32)
        else:
            if means is None and n_components is None:
                raise ValueError("If preset=False, provide `means` or `n_components`.")
            if means is None:
                means = torch.zeros((int(n_components), self.dim), dtype=torch.float32)
            self.means = torch.as_tensor(means, dtype=torch.float32)
            if self.means.ndim != 2 or self.means.shape[1] != self.dim:
                raise ValueError(
                    f"`means` must have shape (n_components, {self.dim}), got {tuple(self.means.shape)}"
                )
            self.n_components = int(self.means.shape[0])

            if stds is None:
                stds = 0.05 if self.dim == 1 else 0.2

            if shared_std:
                std_scalar = float(stds)
                self.stds = torch.full((self.n_components, self.dim), std_scalar, dtype=torch.float32)
            else:
                stds_tensor = torch.as_tensor(stds, dtype=torch.float32)
                if stds_tensor.shape != (self.n_components, self.dim):
                    raise ValueError(
                        f"If shared_std=False, `stds` must have shape ({self.n_components}, {self.dim}), "
                        f"got {tuple(stds_tensor.shape)}"
                    )
                self.stds = stds_tensor

        # Uniform mixture assignment.
        self.labels = torch.randint(low=0, high=self.n_components, size=(self.n,))
        means_per_sample = self.means[self.labels]
        stds_per_sample = self.stds[self.labels]
        self.data = means_per_sample + stds_per_sample * torch.randn((self.n, self.dim), dtype=torch.float32)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int:
                Dataset size.
        """
        return self.n

    def __getitem__(self, idx: int):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                x (torch.Tensor):
                    Sample of shape (dim,).
                y (torch.Tensor):
                    Integer component index corresponding to the sampled Gaussian.
        """
        return self.data[idx], self.labels[idx]