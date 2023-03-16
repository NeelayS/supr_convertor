import torch

from supr_convertor.losses import edge_loss, vertex_loss


def test_vertex_loss():

    estimated_vertices = torch.rand(1, 100, 3)
    target_vertices = torch.rand(1, 100, 3)

    loss = vertex_loss(estimated_vertices, target_vertices, reduction="mean")
    assert loss.ndim == 0

    loss = vertex_loss(estimated_vertices, target_vertices, reduction="sum")
    assert loss.ndim == 0

    weights = torch.rand(100, 3)
    loss = vertex_loss(estimated_vertices, target_vertices)
    assert loss.ndim == 0

    loss = vertex_loss(target_vertices, target_vertices)
    assert loss == 0


def test_edge_loss():

    estimated_vertices = torch.rand(1, 4, 3)
    target_vertices = torch.rand(1, 4, 3)

    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1]])

    loss = edge_loss(
        estimated_vertices, target_vertices, faces=faces, reduction="mean", norm="l1"
    )
    assert loss.ndim == 0

    loss = edge_loss(
        estimated_vertices, target_vertices, faces=faces, reduction="sum", norm="l1"
    )
    assert loss.ndim == 0

    loss = edge_loss(
        estimated_vertices, target_vertices, faces=faces, reduction="mean", norm="l2"
    )
    assert loss.ndim == 0