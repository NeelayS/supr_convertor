import torch

from supr_convertor.config import get_cfg
from supr_convertor.convertor import Convertor

from .utils import get_mock_body_model, get_mock_dataloader


DEVICE = torch.device("cpu") # torch.device("cuda")

def test_Convertor():

    dataloader = get_mock_dataloader(100, 20, 4, 2)
    body_model = get_mock_body_model(100, 2)

    cfg = get_cfg("./tests/configs/basic_test.yaml")
    out_dir = "./tmp"

    convertor = Convertor(cfg, body_model, dataloader, out_dir=out_dir, device=DEVICE)
    convertor.convert()