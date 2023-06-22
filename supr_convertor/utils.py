import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.backends.cudnn as cudnn


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def validate_device(device: str):

    try:
        device = torch.device(device)
        _ = torch.tensor([1.0]).to(device)
        print(f"Using device: {device}")
    except:
        device = torch.device("cpu")
        print("Device is either invalid or not available. Using CPU.")

    return device


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


def deform_vertices(deformation_matrix, vertices):
    return torch.einsum("mn,bni->bmi", [deformation_matrix, vertices])


def read_deformation_matrix(deformation_matrix_path, device=torch.device("cpu")):

    assert os.path.exists(deformation_matrix_path), (
        "Deformation matrix path does not exist:" f" {deformation_matrix_path}"
    )
    with open(deformation_matrix_path, "rb") as f:
        def_transfer_setup = pickle.load(f, encoding="latin1")

    if "mtx" in def_transfer_setup:
        def_matrix = def_transfer_setup["mtx"]

        if hasattr(def_matrix, "todense"):
            def_matrix = def_matrix.todense()

        def_matrix = np.array(def_matrix, dtype=np.float32)

        num_verts = def_matrix.shape[1] // 2
        def_matrix = def_matrix[:, :num_verts]

    elif "matrix" in def_transfer_setup:
        def_matrix = def_transfer_setup["matrix"]

    else:
        valid_keys = ["mtx", "matrix"]
        raise KeyError(f"Deformation matrix setup must contain {valid_keys}")

    def_matrix = torch.tensor(def_matrix, device=device, dtype=torch.float32)

    return def_matrix
