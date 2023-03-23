import os
from typing import Optional, Tuple

import torch
import trimesh


class MeshFolderDataset(torch.utils.data.Dataset):
    """
    Dataloader to load meshes from a folder

    Parameters
    ----------
    data_folder: str
        Path to the folder containing the meshes to load
    exts: Optional[Tuple]
        List of extensions to load. Default: [".obj", ".ply"]
    """

    def __init__(self, data_folder: str, exts: Optional[Tuple] = None):

        if exts is None:
            exts = [".obj", ".ply"]

        self.data_folder = os.path.expandvars(data_folder)

        self.data_paths = sorted(
            [
                os.path.join(self.data_folder, fname)
                for fname in os.listdir(self.data_folder)
                if any(fname.endswith(ext) for ext in exts)
            ],
        )
        self.num_items = len(self.data_paths)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):

        mesh_path = self.data_paths[index]
        mesh = trimesh.load(mesh_path, process=False)

        return {
            "vertices": torch.tensor(mesh.vertices, dtype=torch.float32),
            "faces": torch.tensor(mesh.faces),
        }
