import shutil
from os import path as osp

import msgspec
from traiNNer.data.paired_video_dataset import PairedVideoDataset
from traiNNer.utils.redux_options import DatasetOptions


def test_pairedvideodataset() -> None:
    """Test dataset: PairedVideoDataset"""

    clip_size = 5
    gt_size = 128
    scale = 2

    opt_str = rf"""
name: Test
type: PairedVideoDataset
dataroot_gt: [datasets/train/video/hr]
dataroot_lq: [datasets/train/video/lr]
filename_tmpl: '{{}}'
io_backend:
    type: disk
clip_size: {clip_size}
scale: {scale}
gt_size: {gt_size}
use_hflip: true
use_rot: true

phase: train
"""
    image_names = [f"show1_Frame{i}" for i in range(200, 211)]
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)

    dataset = PairedVideoDataset(opt)
    assert dataset.io_backend_opt["type"] == "disk"  # io backend
    assert len(dataset) == len(image_names) - clip_size + 1  # sliding windows

    # ------------------ test scan folder mode -------------------- #
    opt.io_backend = {"type": "disk"}
    dataset = PairedVideoDataset(opt)
    assert dataset.io_backend_opt["type"] == "disk"  # io backend
    assert len(dataset) == len(image_names) - clip_size + 1  # sliding windows

    expected_middle_frames = image_names[2 : 2 + len(dataset)]

    # test __getitem__
    for i, middle_frame in enumerate(expected_middle_frames):
        result = dataset.__getitem__(i)
        # check returned keys
        expected_keys = ["lq", "gt", "lq_path", "gt_path"]
        assert set(expected_keys).issubset(set(result.keys()))
        # check shape and contents
        assert (
            "gt" in result
            and "lq" in result
            and "lq_path" in result
            and "gt_path" in result
        )
        assert result["gt"].shape == (3, gt_size, gt_size)
        assert result["lq"].shape == (clip_size, 3, gt_size // scale, gt_size // scale)
        print(i, result["lq_path"], result["gt_path"])
        assert osp.normpath(result["lq_path"]) == osp.normpath(
            f"datasets/train/video/lr/{middle_frame}.png"
        )
        assert osp.normpath(result["gt_path"]) == osp.normpath(
            f"datasets/train/video/hr/{middle_frame}.png"
        )


def test_pairedvideodataset_rectangular_crop() -> None:
    clip_size = 5
    gt_size = (128, 96)
    scale = 2

    opt_str = rf"""
name: TestRect
type: PairedVideoDataset
dataroot_gt: [datasets/train/video/hr]
dataroot_lq: [datasets/train/video/lr]
filename_tmpl: '{{}}'
io_backend:
    type: disk
clip_size: {clip_size}
scale: {scale}
gt_size: [{gt_size[0]}, {gt_size[1]}]
use_hflip: true
use_rot: true

phase: train
"""
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)
    dataset = PairedVideoDataset(opt)

    result = dataset.__getitem__(0)
    assert result["gt"].shape == (3, gt_size[0], gt_size[1])
    assert result["lq"].shape == (
        clip_size,
        3,
        gt_size[0] // scale,
        gt_size[1] // scale,
    )


# def test_getitem() -> None:
#     opt = DatasetOptions(
#         name="train",
#         type="train",
#         clip_size=5,
#         num_worker_per_gpu=0,
#         persistent_workers=False,
#         prefetch_factor=None,
#         scale=2,
#         gt_size=128,
#         dataroot_gt=[
#             "datasets/train/send/HR1",
#             "datasets/train/send/HR2",
#             "datasets/train/video/hr",
#         ],
#         dataroot_lq=[
#             "datasets/train/send/LR1",
#             "datasets/train/send/LR2",
#             "datasets/train/video/lr",
#         ],
#     )

#     dataset = PairedVideoDataset(opt)

#     # test __getitem__
#     num_seq = 0
#     assert isinstance(opt.dataroot_gt, list)
#     assert opt.clip_size is not None
#     for p in opt.dataroot_gt:
#         num_seq += len(list(os.listdir(p))) - opt.clip_size + 1

#     print(num_seq)
#     assert num_seq == len(dataset)
#     for i in range(len(dataset)):
#         result = dataset.__getitem__(i)
#         assert "lq_path" in result
#         print(i, result["lq_path"])


def test_pairedvideodataset_uses_alphabetical_sliding_windows(tmp_path) -> None:
    clip_size = 4
    source_lr = tmp_path / "lr"
    source_hr = tmp_path / "hr"
    source_lr.mkdir()
    source_hr.mkdir()

    renamed_files = [
        "beta.png",
        "kappa.png",
        "alpha.png",
        "theta.png",
        "delta.png",
        "omega.png",
        "gamma.png",
        "sigma.png",
        "tau.png",
        "zeta.png",
        "eta.png",
    ]
    source_files = [
        f"datasets/train/video/lr/show1_Frame{frame}.png" for frame in range(200, 211)
    ]

    for src, renamed in zip(source_files, renamed_files, strict=True):
        shutil.copy(src, source_lr / renamed)
        shutil.copy(src.replace("/lr/", "/hr/"), source_hr / renamed)

    opt_str = rf"""
name: AlphabeticalSlidingWindows
type: PairedVideoDataset
dataroot_gt: [{source_hr}]
dataroot_lq: [{source_lr}]
filename_tmpl: '{{}}'
io_backend:
    type: disk
clip_size: {clip_size}
scale: 2
use_hflip: false
use_rot: false

phase: val
"""
    opt = msgspec.yaml.decode(opt_str, type=DatasetOptions, strict=True)
    dataset = PairedVideoDataset(opt)

    assert len(dataset) == len(renamed_files) - clip_size + 1

    alphabetical_files = sorted(renamed_files)
    expected_middle_frames = alphabetical_files[
        clip_size // 2 : clip_size // 2 + len(dataset)
    ]

    for i, middle_frame in enumerate(expected_middle_frames):
        result = dataset[i]
        assert osp.basename(result["lq_path"]) == middle_frame
        assert osp.basename(result["gt_path"]) == middle_frame
