import torch

from supr_convertor.config import get_cfg
from supr_convertor.convertor import Convertor

from .utils import get_mock_body_model, get_mock_dataloader

DEVICE = torch.device("cuda")  # torch.device("cuda")


def test_Convertor():

    cfg = get_cfg("./tests/configs/basic_test.yaml")
    out_dir = "./tmp"

    body_model = get_mock_body_model(100, 1, n_betas=cfg.model.n_betas, device=DEVICE)
    dataloader = get_mock_dataloader(100, 20, 4, 1)
    convertor = Convertor(cfg, body_model, dataloader, out_dir=out_dir, device=DEVICE)
    convertor.convert()

    body_model = get_mock_body_model(100, 2, n_betas=cfg.model.n_betas, device=DEVICE)
    dataloader = get_mock_dataloader(100, 20, 4, 2)
    convertor = Convertor(cfg, body_model, dataloader, out_dir=out_dir, device=DEVICE)
    convertor.convert()

    body_model = get_mock_body_model(100, 100, n_betas=cfg.model.n_betas, device=DEVICE)
    dataloader = get_mock_dataloader(100, 20, 200, 100)
    convertor = Convertor(cfg, body_model, dataloader, out_dir=out_dir, device=DEVICE)
    convertor.convert()
