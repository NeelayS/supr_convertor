import torch

from supr_convertor.utils import get_vertices_per_edge


def vertex_loss(estimated_vertices, target_vertices, reduction="mean"):

    reduction = reduction.lower()
    assert reduction in ("mean", "sum"), "Reduction must be either 'mean' or 'sum'"

    return torch.nn.functional.mse_loss(estimated_vertices, target_vertices, reduction=reduction)


def _compute_edges(vertices, connections):

    if not isinstance(connections, torch.Tensor):
        connections = torch.tensor(connections, device=vertices.device)

    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)

    edges = torch.index_select(vertices, 1, connections.view(-1)).reshape(
        vertices.shape[0], -1, 2, 3
    )

    return edges[:, :, 1] - edges[:, :, 0]


def edge_loss(
    estimated_vertices,
    target_vertices,
    vertices_per_edge=None,
    faces=None,
    reduction="mean",
    norm="l1",
):

    reduction = reduction.lower()
    assert reduction in ("mean", "sum"), "Reduction must be either 'mean' or 'sum'"

    norm = norm.lower()
    assert norm in ("l1", "l2"), "Norm must be either 'l1' or 'l2'"

    if vertices_per_edge is None:
        assert (
            faces is not None
        ), "If vertices_per_edge is not provided, faces must be provided"
        if target_vertices.ndim == 2:
            n_vertices = target_vertices.shape[0]
        else:
            n_vertices = target_vertices.shape[1]
        vertices_per_edge = get_vertices_per_edge(n_vertices, faces.cpu())
        vertices_per_edge = torch.tensor(
            vertices_per_edge, device=estimated_vertices.device
        )

    estimated_edges = _compute_edges(estimated_vertices, vertices_per_edge)
    target_edges = _compute_edges(target_vertices, vertices_per_edge)

    loss = estimated_edges - target_edges

    if norm == "l2":
        loss = loss**2

    else:
        loss = torch.abs(loss)

    if reduction == "sum":
        loss = loss.sum()

    return loss.mean()