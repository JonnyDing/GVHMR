"""Inference API for GVHMR runtime without demo-script dependency."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_module
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.net_utils import detach_to_cpu
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh


def build_cfg(
    video_path: Path,
    output_root: Path,
    static_cam: bool = False,
    use_dpvo: bool = False,
    f_mm: int | None = None,
    verbose: bool = False,
):
    video_path = video_path.resolve()
    output_root = output_root.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
            f"use_dpvo={use_dpvo}",
            f"output_root={output_root}",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    dst_video = Path(cfg.video_path)
    if not dst_video.exists() or get_video_lwh(dst_video)[0] != length:
        Log.info(f"[Copy Video] {video_path} -> {dst_video}")
        shutil.copy2(video_path, dst_video)

    return cfg


@torch.no_grad()
def run_preprocess(cfg) -> None:
    Log.info("[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths

    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")

    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")

    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    if not cfg.static_cam:
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(
                    cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm
                )
                vo_results = simple_vo.compute()
                torch.save(vo_results, paths.slam)
            else:
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(
                    video_path, width, height, intrinsics, buffer=4000, resize=0.5
                )
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        r_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            r_w2c = quaternion_to_matrix(traj_quat).mT
        else:
            r_w2c = torch.from_numpy(traj[:, :3, :3])

    if cfg.f_mm is not None:
        k_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(
            length, 1, 1
        )
    else:
        k_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": k_fullimg,
        "cam_angvel": compute_cam_angvel(r_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def infer_video_to_pt(
    video_path: Path,
    output_root: Path,
    static_cam: bool = False,
    use_dpvo: bool = False,
    f_mm: int | None = None,
) -> Path:
    cfg = build_cfg(
        video_path=video_path,
        output_root=output_root,
        static_cam=static_cam,
        use_dpvo=use_dpvo,
        f_mm=f_mm,
        verbose=False,
    )

    hmr4d_results = Path(cfg.paths.hmr4d_results)
    if hmr4d_results.exists():
        Log.info(f"[HMR4D] Reusing existing result at {hmr4d_results}")
        return hmr4d_results

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GVHMR inference")

    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    Log.info("[HMR4D] Predicting")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    tic = Log.sync_time()
    pred = model.predict(data, static_cam=cfg.static_cam)
    pred = detach_to_cpu(pred)
    data_time = data["length"] / 30
    Log.info(
        f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s"
    )
    torch.save(pred, hmr4d_results)
    return hmr4d_results
