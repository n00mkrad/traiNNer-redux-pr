# Based on FDAT: https://github.com/stinkybread/fdat/blob/main/fdat.py
# ruff: noqa
# type: ignore
import torch
import torch.nn.functional as F  # noqa: N812
from spandrel.__helpers.model_descriptor import StateDict
from spandrel.util.timm import DropPath
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.module import _IncompatibleKeys  # type: ignore

from traiNNer.archs.arch_util import SampleMods3, UniUpsampleV3
from traiNNer.utils.registry import ARCH_REGISTRY


# --- Lightweight Optical Flow Network ---
class LightFlowNet(nn.Module):
    """Ultra-lightweight optical flow estimation for frame alignment.

    Estimates flow from source frame to target frame using a simple encoder.
    TensorRT and ONNX compatible.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        # Simple encoder: 3 conv layers with stride-2 downsampling
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch * 2, base_ch, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Flow predictor at 1/8 resolution
        self.flow_pred = nn.Conv2d(base_ch * 4, 2, 3, 1, 1)
        # Initialize flow prediction to zero
        nn.init.zeros_(self.flow_pred.weight)
        nn.init.zeros_(self.flow_pred.bias)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """Estimate flow from src to tgt.

        Args:
            src: Source frame (B, C, H, W)
            tgt: Target frame (B, C, H, W)

        Returns:
            flow: Optical flow (B, 2, H, W) in pixel units
        """
        B, C, H, W = src.shape
        x = torch.cat([src, tgt], dim=1)
        feat = self.enc(x)
        flow_low = self.flow_pred(feat)
        # Upsample flow to full resolution (scale by 8 for pixel displacement)
        flow = (
            F.interpolate(flow_low, size=(H, W), mode="bilinear", align_corners=False)
            * 8.0
        )
        return flow


def warp_frame(x: Tensor, flow: Tensor, training: bool = False) -> Tensor:
    """Warp frame x according to optical flow."""
    B, C, H, W = x.shape
    
    # Fast path for training: Uses highly optimized native C++ cuDNN kernels
    if training:
        # Create meshgrid mapping to [-1, 1] for grid_sample
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        # Add flow and normalize coordinates to [-1, 1] range
        grid_x = 2.0 * (xx + flow[:, 0]) / max(W - 1, 1) - 1.0
        grid_y = 2.0 * (yy + flow[:, 1]) / max(H - 1, 1) - 1.0
        
        grid = torch.stack((grid_x, grid_y), dim=-1) # (B, H, W, 2)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Slow path for ONNX / DirectML inference
    # ... (Keep your exact existing gather-based bilinear code below) ...
    yy = (
        torch.arange(H, device=x.device, dtype=x.dtype)
        .view(1, 1, H, 1)
        .expand(B, 1, H, W)
    )
    xx = (
        torch.arange(W, device=x.device, dtype=x.dtype)
        .view(1, 1, 1, W)
        .expand(B, 1, H, W)
    )

    # Add flow to get sampling coordinates
    new_x = xx + flow[:, 0:1]
    new_y = yy + flow[:, 1:2]

    # Create max bounds as tensors on the same device for ONNX compatibility
    max_x = torch.tensor(W - 1, device=x.device, dtype=x.dtype)
    max_y = torch.tensor(H - 1, device=x.device, dtype=x.dtype)

    # Clamp coordinates to valid range (border padding behavior)
    new_x = new_x.clamp(0, max_x)
    new_y = new_y.clamp(0, max_y)

    # Get integer coordinates for the 4 neighboring pixels
    x0 = new_x.floor()
    y0 = new_y.floor()
    x1 = (x0 + 1).clamp(max=max_x)
    y1 = (y0 + 1).clamp(max=max_y)

    # Compute interpolation weights
    dx = new_x - x0
    dy = new_y - y0

    # Convert to long for indexing
    x0_l = x0.long().squeeze(1)  # (B, H, W)
    y0_l = y0.long().squeeze(1)
    x1_l = x1.long().squeeze(1)
    y1_l = y1.long().squeeze(1)

    # Compute linear indices for gather operation
    # Flatten spatial dimensions of input
    x_flat = x.view(B, C, -1)  # (B, C, H*W)

    # Compute indices for 4 corners
    idx_00 = (y0_l * W + x0_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)
    idx_01 = (y0_l * W + x1_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)
    idx_10 = (y1_l * W + x0_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)
    idx_11 = (y1_l * W + x1_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)

    # Gather pixel values from 4 corners
    p00 = x_flat.gather(-1, idx_00).view(B, C, H, W)
    p01 = x_flat.gather(-1, idx_01).view(B, C, H, W)
    p10 = x_flat.gather(-1, idx_10).view(B, C, H, W)
    p11 = x_flat.gather(-1, idx_11).view(B, C, H, W)

    # Bilinear interpolation weights
    w00 = (1 - dx) * (1 - dy)
    w01 = dx * (1 - dy)
    w10 = (1 - dx) * dy
    w11 = dx * dy

    # Weighted sum
    return w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11


class FieldPrefill(nn.Module):
    """Prefill missing field lines without altering valid scanlines."""

    def __init__(self, num_in_ch: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_in_ch + 1, hidden, (5, 1), 1, (2, 0), bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, num_in_ch, (3, 1), 1, (1, 0), bias=True),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        delta = self.net(torch.cat([x, mask], dim=1))
        return x + (1.0 - mask) * delta


# --- FDAT Components ---
class FastSpatialWindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.dim, self.ws, self.nh = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv, self.proj = (
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            nn.Linear(dim, dim),
        )
        self.bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )
        trunc_normal_(self.bias, std=0.02)

    def forward(self, x, H, W):
        B, L, C = x.shape

        # Calculate padded dimensions explicitly for TensorRT compatibility
        H_pad = ((H + self.ws - 1) // self.ws) * self.ws
        W_pad = ((W + self.ws - 1) // self.ws) * self.ws
        pad_b = H_pad - H
        pad_r = W_pad - W

        # Reshape to spatial, pad, then process
        x = x.view(B, H, W, C)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        # Partition into windows with explicit window counts
        nH, nW = H_pad // self.ws, W_pad // self.ws
        x = (
            x.view(B, nH, self.ws, nW, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B * nH * nW, self.ws * self.ws, C)
        )

        # Attention
        qkv = (
            self.qkv(x)
            .view(-1, self.ws * self.ws, 3, self.nh, C // self.nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale @ k.transpose(-2, -1)) + self.bias
        x = (
            (F.softmax(attn, dim=-1) @ v)
            .transpose(1, 2)
            .reshape(-1, self.ws * self.ws, C)
        )

        # Reverse window partition
        x = (
            self.proj(x)
            .view(B, nH, nW, self.ws, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, H_pad, W_pad, C)
        )

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        return x.view(B, H * W, C)


class FastChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.nh = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv, self.proj = (
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            nn.Linear(dim, dim),
        )

    def forward(self, x, H, W):  # H, W are unused but kept for API consistency
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = (
            F.normalize(q.transpose(-2, -1), dim=-1),
            F.normalize(k.transpose(-2, -1), dim=-1),
        )
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temp, dim=-1)
        return self.proj(
            (attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        )


class SimplifiedAIM(nn.Module):
    def __init__(self, dim, reduction_ratio=8) -> None:
        super().__init__()
        self.sg = nn.Sequential(nn.Conv2d(dim, 1, 1, bias=False), nn.Sigmoid())
        self.cg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, attn_feat, conv_feat, interaction_type, H, W):
        B, L, C = attn_feat.shape
        if interaction_type == "spatial_modulates_channel":
            sm = (
                self.sg(attn_feat.transpose(1, 2).view(B, C, H, W))
                .view(B, 1, L)
                .transpose(1, 2)
            )
            return attn_feat + (conv_feat * sm)
        else:
            cm = (
                self.cg(conv_feat.transpose(1, 2).view(B, C, H, W))
                .view(B, C, 1)
                .transpose(1, 2)
            )
            return (attn_feat * cm) + conv_feat


class SimplifiedFFN(nn.Module):
    def __init__(self, dim, expansion_ratio=2.0, drop=0.0) -> None:
        super().__init__()
        hd = int(dim * expansion_ratio)
        self.fc1, self.act, self.fc2 = (
            nn.Linear(dim, hd, False),
            nn.GELU(),
            nn.Linear(hd, dim, False),
        )
        self.drop = nn.Dropout(drop)
        self.smix = nn.Conv2d(hd, hd, 3, 1, 1, groups=hd, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.drop(self.act(self.fc1(x)))
        x_s = (
            self.smix(x.transpose(1, 2).view(B, x.shape[-1], H, W))
            .view(B, x.shape[-1], L)
            .transpose(1, 2)
        )
        return self.drop(self.fc2(x_s))


class SimplifiedDATBlock(nn.Module):
    def __init__(self, dim, nh, ws, ffn_exp, aim_re, btype, dp, qkv_b=False) -> None:
        super().__init__()
        self.btype = btype
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = (
            FastSpatialWindowAttention(dim, ws, nh, qkv_b)
            if btype == "spatial"
            else FastChannelAttention(dim, nh, qkv_b)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.GELU()
        )
        self.inter = SimplifiedAIM(dim, aim_re)
        self.dp = DropPath(dp) if dp > 0.0 else nn.Identity()
        self.ffn = SimplifiedFFN(dim, ffn_exp)

    def _conv_fwd(self, x, H, W):
        B, L, C = x.shape
        return (
            self.conv(x.transpose(1, 2).view(B, C, H, W)).view(B, C, L).transpose(1, 2)
        )

    def forward(self, x, H, W):
        n1 = self.n1(x)
        itype = (
            "channel_modulates_spatial"
            if self.btype == "spatial"
            else "spatial_modulates_channel"
        )
        fused = self.inter(self.attn(n1, H, W), self._conv_fwd(n1, H, W), itype, H, W)
        x = x + self.dp(fused)
        x = x + self.dp(self.ffn(self.n2(x), H, W))
        return x


class SimplifiedResidualGroup(nn.Module):
    def __init__(self, dim, depth, nh, ws, ffn_exp, aim_re, pattern, dp_rates) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SimplifiedDATBlock(
                    dim, nh, ws, ffn_exp, aim_re, pattern[i % len(pattern)], dp_rates[i]
                )
                for i in range(depth)
            ]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_seq = x.view(B, C, H * W).transpose(1, 2).contiguous()
        for block in self.blocks:
            x_seq = block(x_seq, H, W)
        return self.conv(x_seq.transpose(1, 2).view(B, C, H, W)) + x


class TFDAT(nn.Module):
    """Temporal FDAT for Video Super-Resolution.

    Input: (B, T, C, H, W) - batch of video clips with T frames (T must be odd)
    Output: (B, C, H*scale, W*scale) - upscaled center frame

    Compatible with FDAT pretrained weights (spatial backbone).
    TensorRT and ONNX compatible.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 2,
        clip_size: int = 5,
        embed_dim: int = 64,
        num_groups: int = 4,
        depth_per_group: int = 3,
        num_heads: int = 4,
        window_size: int = 8,
        ffn_expansion_ratio: float = 2.0,
        aim_reduction_ratio: int = 8,
        group_block_pattern: list[str] | None = None,
        drop_path_rate: float = 0.1,
        mid_dim: int = 64,
        upsampler_type: SampleMods3 = "pixelshuffle",
        flow_base_ch: int = 32,
        enable_field_prefill: bool = False,
    ) -> None:
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel"]
        if clip_size % 2 == 0:
            raise ValueError("clip_size must be odd")
        super().__init__()

        self.clip_size = clip_size
        self.upscale = scale
        self.num_in_ch = num_in_ch
        self.enable_field_prefill = enable_field_prefill

        # Unshuffle setup for 1x and 2x scale
        self.unshuffle = 1
        internal_scale = scale
        if scale <= 2:
            self.unshuffle = 4 // max(scale, 1) if scale > 0 else 4
            internal_scale = 4
            conv_in_ch = num_in_ch * self.unshuffle**2
        else:
            conv_in_ch = num_in_ch

        # Lightweight optical flow network (operates on original resolution)
        self.flow_net = LightFlowNet(num_in_ch, flow_base_ch)

        # Optional field prefill for sparse field-based inputs.
        # Keep the module instantiated so checkpoints can be loaded with strict=False
        # regardless of whether this path is enabled for a particular run.
        self.field_prefill = FieldPrefill(num_in_ch, 32)

        # Shallow feature extraction stem for scale > 2 (where unshuffle=1)
        # This compensates for the reduced input channel count
        if self.unshuffle == 1:
            self.shallow_stem = nn.Sequential(
                nn.Conv2d(num_in_ch, embed_dim // 2, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 2, num_in_ch, 3, 1, 1, bias=True),
            )
        else:
            self.shallow_stem = None

        # Feature extraction (after potential unshuffle or shallow stem)
        self.conv_first = nn.Conv2d(conv_in_ch, embed_dim, 3, 1, 1, bias=True)

        # Temporal fusion: combine aligned features with center frame features
        # Input: center_feat + sum of (T-1) aligned neighbor features
        self.temporal_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True),
        )

        # Main FDAT backbone
        ad = depth_per_group * len(group_block_pattern)
        td = num_groups * ad
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, td)]

        self.groups = nn.Sequential(
            *[
                SimplifiedResidualGroup(
                    embed_dim,
                    ad,
                    num_heads,
                    window_size,
                    ffn_expansion_ratio,
                    aim_reduction_ratio,
                    group_block_pattern,
                    dpr[i * ad : (i + 1) * ad],
                )
                for i in range(num_groups)
            ]
        )

        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = UniUpsampleV3(
            upsampler_type, internal_scale, embed_dim, num_out_ch, mid_dim, 4
        )

        # Compute alignment requirement: LCM of (unshuffle * window_size) and flow network (8)
        # Flow network has 3 stride-2 convs, so needs divisibility by 8
        # After unshuffle, feature dims must still be divisible by window_size,
        # so input must be divisible by (unshuffle * window_size)
        from math import gcd

        def lcm(a, b):
            return a * b // gcd(a, b)

        self.align = lcm(self.unshuffle * window_size, 8)

        self.apply(self._init_weights)

        # Special initialization for shallow_stem (after general init)
        # Initialize to near-zero for smooth residual learning
        if self.shallow_stem is not None:
            nn.init.zeros_(self.shallow_stem[-1].bias)
            with torch.no_grad():
                self.shallow_stem[-1].weight.mul_(0.1)

        # Initialize field prefill to an exact no-op at startup
        nn.init.zeros_(self.field_prefill.net[-1].weight)
        nn.init.zeros_(self.field_prefill.net[-1].bias)

    def load_state_dict(
        self,
        state_dict: StateDict,
        strict: bool = False,
        *args,
        **kwargs,
    ) -> _IncompatibleKeys:
        state_dict["upsampler.MetaUpsample"] = self.upsampler.MetaUpsample
        # Always use strict=False to allow loading FDAT pretrained weights
        return super().load_state_dict(state_dict, strict=False, *args, **kwargs)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm | nn.GroupNorm):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, T, C, H, W)

        Returns:
            Upscaled center frame (B, C, H*scale, W*scale)
        """
        B, T, C, H, W = x.shape
        center_idx = T // 2

        # Compute padding for alignment (must satisfy flow net, window attention, and unshuffle)
        pad_h = (self.align - H % self.align) % self.align
        pad_w = (self.align - W % self.align) % self.align
        if pad_h > 0 or pad_w > 0:
            # Reshape to 4D for padding (reflect mode doesn't support 5D with pad size 4)
            x = x.view(B * T, C, H, W)
            # Use replicate if padding >= dimension size (reflect requires padding < dim)
            pad_mode = "reflect" if pad_h < H and pad_w < W else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            x = x.view(B, T, C, H + pad_h, W + pad_w)
        H_pad, W_pad = H + pad_h, W + pad_w

        if self.enable_field_prefill:
            # Estimate which raster parity contains valid lines in each field
            even_energy = x[..., 0::2, :].abs().mean(dim=(2, 3, 4), keepdim=True)
            odd_energy = x[..., 1::2, :].abs().mean(dim=(2, 3, 4), keepdim=True)
            even_valid = (even_energy >= odd_energy).to(x.dtype)

            rows = torch.arange(H_pad, device=x.device).view(1, 1, 1, H_pad, 1)
            even_rows = (rows.remainder(2) == 0).to(x.dtype)
            mask = even_valid * even_rows + (1.0 - even_valid) * (1.0 - even_rows)
            mask = mask.expand(B, T, 1, H_pad, W_pad)

            # Prefill missing field lines for all frames before motion estimation
            x_prefill = self.field_prefill(
                x.view(B * T, C, H_pad, W_pad),
                mask.view(B * T, 1, H_pad, W_pad),
            )
            x = x_prefill.view(B, T, C, H_pad, W_pad)

        # Extract center frame
        center = x[:, center_idx]  # (B, C, H_pad, W_pad)

        # Gather neighbor frames (everything except the center)
        neighbor_indices = [t for t in range(T) if t != center_idx]
        neighbors = x[:, neighbor_indices]  # (B, T-1, C, H_pad, W_pad)
        
        # Flatten Batch and Temporal dimensions into a single pass
        Tm1 = T - 1
        neighbors_flat = neighbors.reshape(B * Tm1, C, H_pad, W_pad)
        
        # Expand center to match the number of neighbors, then flatten
        center_expanded = center.unsqueeze(1).expand(B, Tm1, C, H_pad, W_pad).reshape(B * Tm1, C, H_pad, W_pad)

        # Compute optical flow in ONE pass
        flows_flat = self.flow_net(neighbors_flat, center_expanded)
        
        # Warp all frames in ONE pass
        aligned_flat = warp_frame(neighbors_flat, flows_flat, self.training)
        
        # Reshape back to separate sequence and sum across the temporal dimension
        aligned_sum = aligned_flat.view(B, Tm1, C, H_pad, W_pad).sum(dim=1)

        # Apply unshuffle for scale <= 2, or shallow stem for scale > 2
        if self.unshuffle > 1:
            center = F.pixel_unshuffle(center, self.unshuffle)
            aligned_sum = F.pixel_unshuffle(aligned_sum, self.unshuffle)
            H_feat, W_feat = H_pad // self.unshuffle, W_pad // self.unshuffle
        else:
            # Apply shallow stem to enrich features before main extraction (scale > 2)
            if self.shallow_stem is not None:
                center = self.shallow_stem(center) + center
                aligned_sum = self.shallow_stem(aligned_sum) + aligned_sum
            H_feat, W_feat = H_pad, W_pad

        # Extract features
        center_feat = self.conv_first(center)
        aligned_feat = self.conv_first(aligned_sum / max(T - 1, 1))

        # Temporal fusion
        fused = self.temporal_fuse(torch.cat([center_feat, aligned_feat], dim=1))

        # Main backbone with residual
        x_shallow = fused
        x_deep = self.groups(x_shallow)
        x_deep = self.conv_after(x_deep)
        x_out = self.upsampler(x_deep + x_shallow)

        # Crop to original size (accounting for scale)
        return x_out[:, :, : H * self.upscale, : W * self.upscale]


# --- Model Variants ---

@ARCH_REGISTRY.register()
def tfdat(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    scale: int = 4,
    embed_dim: int = 120,
    num_groups: int = 4,
    depth_per_group: int = 3,
    num_heads: int = 4,
    window_size: int = 8,
    ffn_expansion_ratio: float = 2.0,
    aim_reduction_ratio: int = 8,
    group_block_pattern: list[str] | None = None,
    drop_path_rate: float = 0.1,
    upsampler_type: SampleMods3 = "transpose+conv",
    flow_base_ch: int = 32,
    enable_field_prefill: bool = False,
) -> TFDAT:
    return TFDAT(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=group_block_pattern,
        drop_path_rate=drop_path_rate,
        upsampler_type=upsampler_type,
        flow_base_ch=flow_base_ch,
        enable_field_prefill=enable_field_prefill,
    )