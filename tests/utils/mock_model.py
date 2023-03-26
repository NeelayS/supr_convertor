import torch


class MockBodyModel(torch.nn.Module):
    def __init__(
        self,
        n_vertices,
        batch_size=1,
        n_betas=10,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.n_vertices = n_vertices
        self.batch_size = batch_size
        self.device = device
        self.n_betas = n_betas

        self.layer = torch.nn.Linear(n_betas + 225 + 3, n_vertices * 3)

    def forward(self, betas, pose, trans, *args, **kwargs):

        vertices = self.layer(torch.cat([betas, pose, trans], dim=-1)).view(
            -1, self.n_vertices, 3
        )

        return {
            "vertices": vertices,
        }


def get_mock_body_model(
    n_vertices, batch_size=1, n_betas=10, device=torch.device("cpu")
):
    return MockBodyModel(n_vertices, batch_size, n_betas, device)
