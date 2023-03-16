import torch


class MockBodyModel(torch.nn.Module):
    def __init__(self, n_vertices, batch_size=1):
        super().__init__()

        self.n_vertices = n_vertices
        self.batch_size = batch_size

    def forward(self, *args, **kwargs):

        return {
            "vertices": torch.rand(self.batch_size, self.n_vertices, 3),
            "faces": [],
        }


def get_mock_body_model(n_vertices, batch_size=1):
    return MockBodyModel(n_vertices, batch_size)