from typing import Union

import numpy as np
import torch

from supr_convertor.utils import get_vertices_per_edge


def v2v_error(
    estimated_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    reduction: str = "mean",
):
    """
    Vertex to vertex error in meters.

    Parameters
    ----------
    estimated_vertices: torch.Tensor
        Estimated vertices, shape must be either (batch_size, n_vertices, 3) or (n_vertices, 3)
    target_vertices: torch.Tensor
        Target vertices, shape must be either (batch_size, n_vertices, 3) or (n_vertices, 3)
    reduction: str
        Reduction method. Either "mean" or "sum". Default: "mean"

    Returns
    -------
    error: torch.Tensor
        Error value
    """

    reduction = reduction.lower()
    assert reduction in ("mean", "sum"), "Reduction must be either 'mean' or 'sum'"

    error = torch.sqrt(torch.sum((estimated_vertices - target_vertices) ** 2, axis=-1))

    if reduction == "mean":
        return torch.mean(error)

    error = torch.sum(error, axis=-1)

    return torch.mean(error)


def vertex_loss(
    estimated_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    reduction: str = "mean",
):
    """
    Squared L2 loss between estimated and target vertices.

    Parameters
    ----------
    estimated_vertices: torch.Tensor
        Estimated vertices, shape must be either (batch_size, n_vertices, 3) or (n_vertices, 3)
    target_vertices: torch.Tensor
        Target vertices, shape must be either (batch_size, n_vertices, 3) or (n_vertices, 3)
    reduction: str
        Reduction method. Either "mean" or "sum". Default: "mean"

    Returns
    -------
    loss: torch.Tensor
        Loss value
    """

    reduction = reduction.lower()
    assert reduction in ("mean", "sum"), "Reduction must be either 'mean' or 'sum'"

    loss = torch.sum((estimated_vertices - target_vertices) ** 2, dim=-1)

    if reduction == "mean":
        return torch.mean(loss)

    loss = torch.sum(loss, dim=-1)

    return torch.mean(loss)


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
    estimated_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    vertices_per_edge: torch.tensor = None,
    faces: Union[torch.Tensor, np.ndarray] = None,
    reduction: str = "mean",
    norm: str = "l1",
):
    """
    Edge loss between estimated and target vertices.

    Parameters
    ----------
    estimated_vertices: torch.Tensor
        Estimated vertices, shape must be either (batch_size, n_vertices, 3) or (n_vertices, 3)
    target_vertices: torch.Tensor
        Target vertices, shape must be either (batch_size, n_vertices, 3) or (n_vertices, 3)
    vertices_per_edge: torch.Tensor
        Vertices per edge, shape must be (n_edges, 2)
    faces: torch.Tensor or np.ndarray
        Faces, shape must be (n_faces, 3)
    reduction: str
        Reduction method. Either "mean" or "sum". Default: "mean"
    norm: str
        Norm to use. Either "l1" or "l2". Default: "l1"
    """

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
        vertices_per_edge = get_vertices_per_edge(n_vertices, faces)
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
