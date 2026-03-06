# models/unet.py
import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    U-Net backbone (encoder–decoder with skip connections).

    Typical use:
        - Segmentation (predicting per-pixel classes)
        - Diffusion / denoising (predicting noise or x0), often with timestep conditioning

    This is a template (API + docstrings only). Implementations should define:
        - building blocks (Conv blocks, downsample, upsample)
        - forward pass wiring (skip connections)
        - optional timestep / conditioning injection (for diffusion)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        use_attention: bool = False,
        time_emb_dim: int | None = None,
        num_groups: int = 32,
    ) -> None:
        """
        Args:
            in_channels:
                Number of input channels (e.g., 3 for RGB, 1 for grayscale).
            out_channels:
                Number of output channels (e.g., #classes for segmentation,
                or same as in_channels for diffusion noise prediction).
            base_channels:
                Base feature width for the first stage.
            channel_mults:
                Multipliers applied to base_channels at each downsampling stage.
                Example: base_channels=64 and channel_mults=(1,2,4,8) gives
                channels: 64, 128, 256, 512.
            num_res_blocks:
                Number of conv/residual blocks per stage (encoder and decoder).
            dropout:
                Dropout probability used inside blocks (if enabled).
            use_attention:
                If True, include attention blocks at one or more resolutions.
            time_emb_dim:
                If provided, enables timestep conditioning (diffusion-style).
                The model is expected to embed timesteps to this dimension and
                inject it into blocks.
            num_groups:
                GroupNorm parameter when GroupNorm is used (common in diffusion U-Nets).

        Notes:
            - This template does not assume a specific block type (plain conv vs residual).
            - For diffusion, it is common to use GroupNorm + SiLU, and inject time embeddings
              into residual blocks.
        """
        super().__init__()
        raise NotImplementedError("UNet.__init__ not implemented.")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net.

        Args:
            x:
                Input tensor of shape (B, in_channels, H, W).
            t:
                Optional timestep tensor of shape (B,). Used for diffusion models.
                If time_emb_dim is None, this should be ignored or forbidden.
            cond:
                Optional conditioning tensor (e.g., class embedding, text embedding, etc.).
                How it is used depends on the intended conditioning mechanism.

        Returns:
            Output tensor of shape (B, out_channels, H, W).

        Raises:
            ValueError:
                If t is provided but the model was not configured for timestep conditioning,
                or if shapes are inconsistent.
        """
        raise NotImplementedError("UNet.forward not implemented.")