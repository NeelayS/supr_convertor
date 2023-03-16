import torch

from supr_convertor.utils import v2v_error


def test_v2v_error():

    vertices = torch.rand(100, 3)

    error = v2v_error(vertices, vertices)
    assert error == 0.0
