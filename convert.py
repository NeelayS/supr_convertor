import argparse

import torch

from SUPR.supr.pytorch.supr import SUPR
from supr_convertor.config import get_cfg
from supr_convertor.convertor import Convertor
from supr_convertor.data import MeshFolderDataset
from supr_convertor.utils import validate_device

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = get_cfg(args.cfg)

    print(f"\nUsing {cfg.model.gender} body model")

    device = validate_device(cfg.device)

    dataset = MeshFolderDataset(cfg.data.mesh_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    target_body_model = SUPR(
        cfg.model.path,
        num_betas=cfg.model.n_betas,
        device=device,
    )

    convertor = Convertor(
        cfg,
        target_body_model,
        dataloader,
        device,
    )
    convertor.convert()
