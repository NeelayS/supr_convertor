import torch


class MockMeshDataset(torch.utils.data.Dataset):
    def __init__(self, n_vertices, n_faces, n_samples):

        self.n_vertices = n_vertices
        self.n_faces = n_faces
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        vertices = torch.rand(self.n_vertices, 3)
        faces = torch.zeros(self.n_faces, 3)

        i = self.n_vertices // 3
        faces[:, 0] = torch.randint(0, i, (self.n_faces,))
        faces[:, 1] = torch.randint(i, 2 * i, (self.n_faces,))
        faces[:, 2] = torch.randint(2 * i, 3 * i, (self.n_faces,))

        return {
            "vertices": vertices,
            "faces": faces,
        }


def get_mock_dataloader(n_vertices, n_faces, n_samples, batch_size):

    dataset = MockMeshDataset(n_vertices, n_faces, n_samples)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)