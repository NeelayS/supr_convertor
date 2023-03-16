import numpy as np
import scipy.sparse as sp
import torch
import torch.backends.cudnn as cudnn


def _row(A):
    return A.reshape((1, -1))


def _col(A):
    return A.reshape((-1, 1))


def get_vertex_connectivity(n_vertices, faces):

    vpv = sp.csc_matrix((n_vertices, n_vertices))

    for i in range(3):

        if faces.ndim == 3:
            faces = faces[0]

        IS = faces[:, i]
        JS = faces[:, (i + 1) % 3]

        data = np.ones(len(IS))
        ij = np.vstack((_row(IS.flatten()), _row(JS.flatten())))

        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(n_vertices, faces):

    vc = sp.coo_matrix(get_vertex_connectivity(n_vertices, faces))

    vpe = np.hstack((_col(vc.row), _col(vc.col)))
    vpe = vpe[vpe[:, 0] < vpe[:, 1]]

    return vpe


def v2v_error(estimated_vertices, target_vertices):
    """
    Vertex to vertex error (in m)
    """
    return torch.mean(
        torch.sqrt(torch.sum((estimated_vertices - target_vertices) ** 2.0, axis=-1))
    ).item()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
