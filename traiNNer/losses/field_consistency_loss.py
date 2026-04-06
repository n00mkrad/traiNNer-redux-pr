from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FieldConsistencyLoss(nn.Module):
    """Penalize inconsistency between even and odd interlaced fields.

    The prediction is split into alternating scanlines, each field is upsampled
    back to full frame height, and a reconstruction loss is applied between the
    two field reconstructions.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        criterion: Literal["charbonnier", "l1", "l2"] = "l1",
        upsample_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion == "l2":
            self.criterion = nn.MSELoss()
        elif criterion == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def _upsample(self, x: Tensor) -> Tensor:
        if self.upsample_mode == "nearest":
            return F.interpolate(x, scale_factor=(2, 1), mode=self.upsample_mode)
        return F.interpolate(
            x,
            scale_factor=(2, 1),
            mode=self.upsample_mode,
            align_corners=self.align_corners,
        )

    def forward(self, pred: Tensor, target: Tensor | None = None) -> Tensor:
        if pred.ndim != 4:
            raise ValueError(
                f"Expected pred with shape (N, C, H, W), but got {tuple(pred.shape)}"
            )

        # No field pairs exist when the frame has fewer than 2 lines.
        if pred.shape[2] < 2:
            return pred.new_zeros(())

        even_lines = pred[:, :, 0::2, :]
        odd_lines = pred[:, :, 1::2, :]

        even_up = self._upsample(even_lines)
        odd_up = self._upsample(odd_lines)

        # For odd frame heights, upsampled field heights can differ by one line.
        min_h = min(even_up.shape[2], odd_up.shape[2])
        if min_h == 0:
            return pred.new_zeros(())

        even_up = even_up[:, :, :min_h, :]
        odd_up = odd_up[:, :, :min_h, :]

        return self.criterion(even_up, odd_up)