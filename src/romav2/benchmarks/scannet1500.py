import json
import os
import os.path as osp
import logging
import numpy as np
import torch
from romav2.geometry import compute_pose_error, pose_auc, estimate_pose_cv2_ransac
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ScanNet1500:
    def __init__(self, data_root="data/scannet/scans") -> None:
        self.data_root = data_root

    def benchmark(self, model, model_name=None, max_pairs=None, seed=0, output=None, dump_dir=None):
        np.random.seed(seed)
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)
        model.train(False)
        thresholds = [5, 10, 20]
        data_root = self.data_root
        tmp = np.load(osp.join(data_root, "test.npz"))
        pairs, rel_pose = tmp["name"], tmp["rel_pose"]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
            rel_pose = rel_pose[:max_pairs]
        tot_e_t, tot_e_R, tot_e_pose = [], [], []
        pair_inds = range(len(pairs))
        for pairind in (pbar := tqdm(pair_inds, smoothing=0.9)):
            scene = pairs[pairind]
            scene_name = f"scene0{scene[0]}_00"
            im_A_path = osp.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[2]}.jpg",
            )
            im_A = Image.open(im_A_path)
            im_B_path = osp.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[3]}.jpg",
            )
            im_B = Image.open(im_B_path)
            T_gt = rel_pose[pairind].reshape(3, 4)
            R, t = T_gt[:3, :3], T_gt[:3, 3]
            K = np.stack(
                [
                    np.array([float(i) for i in r.split()])
                    for r in open(
                        osp.join(
                            self.data_root,
                            "scans_test",
                            scene_name,
                            "intrinsic",
                            "intrinsic_color.txt",
                        ),
                        "r",
                    )
                    .read()
                    .split("\n")
                    if r
                ]
            )
            w1, h1 = im_A.size
            w2, h2 = im_B.size
            preds = model.match(im_A_path, im_B_path)
            sparse_matches, _, _, _ = model.sample(preds, 5000)
            if dump_dir is not None:
                torch.save({
                    'sparse_matches': sparse_matches.cpu(),
                    'im_A': np.array(im_A),
                    'im_B': np.array(im_B),
                    'im_A_path': im_A_path,
                    'im_B_path': im_B_path,
                    'W_A': int(im_A.size[0]), 'H_A': int(im_A.size[1]),
                    'W_B': int(im_B.size[0]), 'H_B': int(im_B.size[1]),
                }, osp.join(dump_dir, f'{pairind:05d}.pt'))
            scale1 = 480 / min(w1, h1)
            scale2 = 480 / min(w2, h2)
            w1, h1 = scale1 * w1, scale1 * h1
            w2, h2 = scale2 * w2, scale2 * h2
            K1 = K.copy()
            K2 = K.copy()
            K1[:2] = K1[:2] * scale1
            K2[:2] = K2[:2] * scale2

            offset = 0.5
            kpts1, kpts2 = model.to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
            kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
            kpts1 = kpts1 - offset
            kpts2 = kpts2 - offset

            for _ in range(5):
                shuffling = np.random.permutation(np.arange(len(kpts1)))
                kpts1 = kpts1[shuffling]
                kpts2 = kpts2[shuffling]
                try:
                    norm_threshold = 0.5 / (
                        np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
                    )
                    R_est, t_est, mask = estimate_pose_cv2_ransac(
                        kpts1,
                        kpts2,
                        K1,
                        K2,
                        norm_threshold,
                        conf=0.99999,
                    )
                    e_t, e_R = compute_pose_error(R_est, t_est[:, 0], R, t)
                    e_pose = max(e_t, e_R)
                except Exception as e:
                    logger.debug(f"Pose estimation error: {e}")
                    e_t, e_R = 90, 90
                    e_pose = max(e_t, e_R)
                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)
            pbar.set_postfix(
                auc=f"{[f'{a.item():.3f}' for a in pose_auc(tot_e_pose, thresholds)]}"
            )
        tot_e_pose = np.array(tot_e_pose)
        auc = pose_auc(tot_e_pose, thresholds)
        acc_5 = (tot_e_pose < 5).mean()
        acc_10 = (tot_e_pose < 10).mean()
        acc_15 = (tot_e_pose < 15).mean()
        acc_20 = (tot_e_pose < 20).mean()
        map_5 = acc_5
        map_10 = np.mean([acc_5, acc_10])
        map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
        results = {
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
            "map_5": map_5,
            "map_10": map_10,
            "map_20": map_20,
        }
        if output is not None:
            os.makedirs(osp.dirname(osp.abspath(output)), exist_ok=True)
            with open(output, 'w') as f:
                json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
        return results
