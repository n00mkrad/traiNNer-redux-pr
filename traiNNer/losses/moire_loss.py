import torch
import torch.fft
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MoireFrequencyLoss(nn.Module):
    """Frequency-domain loss targeting vertical high-frequency differences.

    Penalizes differences in vertical high-frequency content between pred and
    target. For de-interlacing models, moiré artifacts arise when the model
    over-applies field interpolation to fine line patterns (e.g. fabric,
    grilles), suppressing vertical high-frequency content that is not actually
    a combing artifact. Training with this loss discourages that over-smoothing
    by making the model preserve vertical HF energy on all training samples.

    The loss computes the 2D rFFT of both pred and target, applies a directional
    mask that emphasizes vertical high-frequency bins (where deinterlacing moiré
    appears), and returns the L1 difference of the masked magnitude spectra.

    Args:
        loss_weight (float): Weight for this loss. Applied externally by the
            training loop; not used inside forward().
        hf_frac (float): Fraction of the vertical frequency spectrum that
            receives full weight. 0.5 means the top 50% of vertical frequencies
            (from half-Nyquist to Nyquist) get weight 1.0. Default: 0.5.
        lf_cutoff (float): Normalized vertical frequency below which the mask
            is zero. Frequencies between lf_cutoff and (1 - hf_frac) ramp
            linearly from 0 to 1. Set to 0.0 to ramp from DC. Default: 0.25.
        weight_power (float): Exponent applied to the ramp region of the mask.
            Higher values concentrate weight toward the very highest vertical
            frequencies. Default: 1.0.
        horiz_atten (float): Horizontal frequency attenuation. When > 0, bins
            with high horizontal frequency are down-weighted, focusing the loss
            on the (high vertical, low horizontal) content where de-interlacing
            artifacts are most visible. 0.0 disables. 1.0 fully attenuates
            horizontal Nyquist. Default: 0.0.
        use_luma (bool): Operate on luminance only (BT.601 luma). Moiré is
            primarily a luminance artifact. Default: True.
    """

    def __init__(
        self,
        loss_weight: float,
        hf_frac: float = 0.5,
        lf_cutoff: float = 0.25,
        weight_power: float = 1.0,
        horiz_atten: float = 0.0,
        use_luma: bool = True,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.hf_frac = hf_frac
        self.lf_cutoff = lf_cutoff
        self.weight_power = weight_power
        self.horiz_atten = horiz_atten
        self.use_luma = use_luma
        self._mask_cache: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    def _to_luma(self, x: Tensor) -> Tensor:
        # BT.601 luma, output shape (N, 1, H, W)
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
        return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    def _build_mask(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        cache_key = (h, w, device, dtype)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        # Absolute normalized vertical frequency: 0.0 at DC, 1.0 at Nyquist.
        vert_freq = torch.fft.fftfreq(h, device=device, dtype=dtype).abs() * 2

        # Full-weight boundary: vertical frequencies above this get weight 1.0.
        hf_boundary = max(1.0 - self.hf_frac, 1e-6)
        lf = self.lf_cutoff

        if lf >= hf_boundary:
            # No ramp region: step from 0 to 1 at boundary.
            vert_mask = (vert_freq >= hf_boundary).to(dtype)
        else:
            # Zero below lf_cutoff, linear ramp to hf_boundary, one above.
            vert_mask = ((vert_freq - lf) / (hf_boundary - lf)).clamp(0.0, 1.0)
            if self.weight_power != 1.0:
                vert_mask = vert_mask ** self.weight_power

        w_rfft = w // 2 + 1
        mask = vert_mask.view(h, 1).expand(h, w_rfft)

        # Down-weight high horizontal frequencies so the loss focuses on
        # (high vertical, low horizontal) bins — where interlacing moiré lives.
        if self.horiz_atten > 0:
            horiz_freq = torch.fft.rfftfreq(w, device=device, dtype=dtype) * 2
            horiz_weight = (1.0 - self.horiz_atten * horiz_freq).clamp(0.0, 1.0)
            mask = mask * horiz_weight.view(1, w_rfft)

        mask = mask.contiguous()
        self._mask_cache[cache_key] = mask
        return mask

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        if self.use_luma and pred.shape[1] == 3:
            pred = self._to_luma(pred)
            target = self._to_luma(target)

        _, _, h, w = pred.shape

        pred_mag = torch.fft.rfft2(pred, norm="ortho").abs()
        target_mag = torch.fft.rfft2(target, norm="ortho").abs()

        mask = self._build_mask(h, w, pred.device, pred.dtype)

        return F.l1_loss(pred_mag * mask, target_mag * mask)
