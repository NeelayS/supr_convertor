import argparse

import torch

from SUPR.supr.pytorch.supr import SUPR
from supr_convertor.config import get_cfg
from supr_convertor.convertor import Convertor
from supr_convertor.data import MeshFolder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = get_cfg(args.cfg)

    print(f"\nUsing {cfg.model.gender} body model")

    DEVICE = "cuda:0"
    device = torch.device(DEVICE)
    print(f"Using device: {DEVICE}")

    dataset = MeshFolder(cfg.data.mesh_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    target_body_model = SUPR(
        "/home/nshah/work/models/supr/supr_female.npy",
        num_betas=cfg.model.n_betas,
        device=device,
    )

    convertor = Convertor(
        cfg,
        target_body_model,
        dataloader,
    )
    convertor.convert()


# with different batch sizes
# with and without edge loss
# with different learning rates
# with different regularization weights
# init params with prev param values
