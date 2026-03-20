import torch

from traiNNer.archs.tfdat_arch import TFDAT, warp_frame


def _reference_tfdat_forward(model: TFDAT, x: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = x.shape
    center_idx = t // 2

    pad_h = (model.align - h % model.align) % model.align
    pad_w = (model.align - w % model.align) % model.align
    if pad_h > 0 or pad_w > 0:
        x = x.view(b * t, c, h, w)
        pad_mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        x = x.view(b, t, c, h + pad_h, w + pad_w)
    h_pad, w_pad = h + pad_h, w + pad_w

    center = x[:, center_idx]
    aligned_sum = torch.zeros_like(center)
    for frame_idx in range(t):
        if frame_idx == center_idx:
            continue
        neighbor = x[:, frame_idx]
        flow = model.flow_net(neighbor, center)
        aligned = warp_frame(
            neighbor,
            flow,
            use_grid_sample=model.use_grid_sample_inference,
        )
        aligned_sum = aligned_sum + aligned

    aligned_ref = aligned_sum / max(t - 1, 1)
    spatial_inputs = torch.cat((center, aligned_ref), dim=0)
    if model.unshuffle > 1:
        spatial_inputs = torch.nn.functional.pixel_unshuffle(
            spatial_inputs, model.unshuffle
        )
    elif model.shallow_stem is not None:
        spatial_inputs = model.shallow_stem(spatial_inputs) + spatial_inputs
    center, aligned_ref = spatial_inputs.chunk(2, dim=0)

    center_feat, aligned_feat = model.conv_first(
        torch.cat((center, aligned_ref), dim=0)
    ).chunk(2, dim=0)
    fused = model.temporal_fuse(torch.cat([center_feat, aligned_feat], dim=1))
    x_shallow = fused
    x_deep = model.groups(x_shallow)
    x_deep = model.conv_after(x_deep)
    x_out = model.upsampler(x_deep + x_shallow)
    return x_out[:, :, : h * model.upscale, : w * model.upscale]


def test_warp_frame_grid_sample_matches_gather() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    flow = torch.randn(2, 2, 8, 8) * 0.25

    gather_out = warp_frame(x, flow, use_grid_sample=False)
    grid_out = warp_frame(x, flow, use_grid_sample=True)

    torch.testing.assert_close(gather_out, grid_out, rtol=1e-4, atol=1e-5)


def test_tfdat_batched_alignment_matches_reference() -> None:
    torch.manual_seed(0)
    model = TFDAT(
        scale=2,
        clip_size=5,
        embed_dim=32,
        num_groups=1,
        depth_per_group=1,
        num_heads=4,
        window_size=4,
        flow_base_ch=16,
        use_grid_sample_train=True,
        use_grid_sample_inference=False,
    ).eval()
    x = torch.randn(1, 5, 3, 16, 16)

    with torch.inference_mode():
        out = model(x)
        ref = _reference_tfdat_forward(model, x)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_tfdat_eval_supports_grid_sample_warp() -> None:
    torch.manual_seed(1)
    model = TFDAT(
        scale=2,
        clip_size=5,
        embed_dim=32,
        num_groups=1,
        depth_per_group=1,
        num_heads=4,
        window_size=4,
        flow_base_ch=16,
        use_grid_sample_inference=True,
    ).eval()
    x = torch.randn(1, 5, 3, 16, 16)

    with torch.inference_mode():
        out = model(x)

    assert out.shape == (1, 3, 32, 32)
