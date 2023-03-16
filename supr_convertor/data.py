import os
from typing import Optional, Tuple

import numpy as np
import trimesh
from torch.utils.data import Dataset


class MeshFolder(Dataset):
    def __init__(self, data_folder: str, transforms=None, exts: Optional[Tuple] = None):

        if exts is None:
            exts = [".obj", ".ply"]

        self.data_folder = os.path.expandvars(data_folder)

        self.data_paths = np.array(
            sorted(
                [
                    os.path.join(self.data_folder, fname)
                    for fname in os.listdir(self.data_folder)
                    if any(fname.endswith(ext) for ext in exts)
                ],
                reverse=False,
            )
        )
        self.num_items = len(self.data_paths)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):

        mesh_path = self.data_paths[index]
        mesh = trimesh.load(mesh_path, process=False)

        return {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32),
            "faces": np.asarray(mesh.faces, dtype=np.int32),
        }
