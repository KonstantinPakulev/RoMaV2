import argparse
import logging

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from romav2 import RoMaV2
from romav2.benchmarks import ScanNet1500

logger = logging.getLogger(__name__)


def test_scannet1500(data_root, output=None, max_pairs=None, seed=0, dump_dir=None):
    model = RoMaV2(RoMaV2.Cfg(compile=False))
    model.apply_setting("scannet1500")
    scannet1500 = ScanNet1500(data_root=data_root)
    res = scannet1500.benchmark(model, max_pairs=max_pairs, seed=seed, output=output, dump_dir=dump_dir)
    if _WANDB:
        wandb.log(res)
    logger.info(f"ScanNet1500 results: {res}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native RoMa v2 ScanNet-1500 evaluation")
    parser.add_argument("--data_root", required=True,
                        help="Path to ScanNet data root (must contain test.npz and scans_test/)")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit to N pairs (debug mode)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--dump_dir", default=None,
                        help="Save per-pair .pt bundles here (for compare_pipeline_outputs.py)")
    args = parser.parse_args()

    res = test_scannet1500(
        data_root=args.data_root,
        output=args.output,
        max_pairs=args.max_pairs,
        seed=args.seed,
        dump_dir=args.dump_dir,
    )
    print(f"\nAUC@5/10/20 = {100*res['auc_5']:.2f} / {100*res['auc_10']:.2f} / {100*res['auc_20']:.2f}")
