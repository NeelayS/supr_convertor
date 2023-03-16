import os
import pickle

import torch
from tqdm import tqdm

from supr_convertor.config import CfgNode
from supr_convertor.losses import edge_loss, vertex_loss
from supr_convertor.utils import seed_everything, v2v_error


class Convertor:
    def __init__(
        self,
        cfg: CfgNode,
        body_model: torch.nn.Module,
        dataloader,
        out_dir: str = None,
        device: torch.device = None,
    ):

        self.cfg = cfg
        self.body_model = body_model
        self.dataloader = dataloader
        self.device = device

        self.out_dir = out_dir if out_dir is not None else self.cfg.out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.params_out_dir = os.path.join(self.out_dir, "params")
        os.makedirs(self.params_out_dir, exist_ok=True)

        if self.cfg.save_output_meshes is True:
            self.meshes_out_dir = os.path.join(self.out_dir, "meshes")
            os.makedirs(self.meshes_out_dir, exist_ok=True)

        self._setup()

    def _setup(self):

        seed_everything(0)

        if self.device is None:
            try:
                self.device = torch.device(self.cfg.experiment.device)
                _ = torch.tensor([1.0]).to(self.device)
                print(f"Using device: {self.device}")
            except:
                self.device = torch.device("cpu")
                print("Device is either invalid or not available. Using CPU.")

        elif type(self.device) == str:
            try:
                self.device = torch.device(self.device)
                _ = torch.tensor([1.0]).to(self.device)
                print(f"Using device: {self.device}")
            except:
                self.device = torch.device("cpu")
                print("Device is either invalid or not available. Using CPU.")

        else:
            assert isinstance(self.device, torch.device)

        self.body_model.to(self.device)
        self.params = {}

    def _init_params(self, batch_size: int, copy_from_prev: bool = False):

        print("Initializing parameters...")

        if copy_from_prev:
            self.params["betas"] = (
                self.params["betas"].clone().detach().requires_grad_(True)
            )
            self.params["pose"] = (
                self.params["pose"].clone().detach().requires_grad_(True)
            )
            self.params["trans"] = (
                self.params["trans"].clone().detach().requires_grad_(True)
            )

        self.params["betas"] = torch.zeros(
            (batch_size, self.cfg.model.n_betas),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.params["pose"] = torch.zeros(
            (batch_size, 75 * 3),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.params["trans"] = torch.zeros(
            (batch_size, 3),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

    def _optimize(
        self,
        target_vertices,
        loss_fn,
        params_to_optimize,
        n_iters,
        regularization_weights=None,
        regularization_iters=None,
        low_loss_threshold=1e-5,
        low_loss_delta_threshold=1e-7,
        apply_rotation_angles_correction=False,
        loss_fn_args={},
    ):

        if regularization_weights is not None:
            assert regularization_iters is not None
            assert (
                len(regularization_weights)
                == len(regularization_iters)
                == len(params_to_optimize)
            )

        optimizer = torch.optim.LBFGS(
            params_to_optimize,
            **self.cfg.experiment.optimizer_params.to_dict(),
        )

        prev_loss = 1e10

        for i in tqdm(range(n_iters)):

            def closure():

                optimizer.zero_grad()
                estimated_vertices = self.body_model(**self.params)["vertices"]
                loss = loss_fn(estimated_vertices, target_vertices, **loss_fn_args)

                if regularization_weights is not None:
                    for param_name, iters in regularization_iters.items():
                        if i < iters:
                            loss += regularization_weights[param_name] * torch.mean(
                                self.params[param_name] ** 2.0
                            )

                loss.backward()

                if self.cfg.experiment.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        params_to_optimize, self.cfg.experiment.grad_clip
                    )

                return loss

            loss = optimizer.step(closure).item()

            if i % 10 == 0:
                print(f"Iteration: {i}, Loss: {loss}")

            if loss < low_loss_threshold:
                print(
                    f"Loss ({loss}) less than the threshold of {low_loss_threshold}. Stopping optimization."
                )
                break

            if abs(prev_loss - loss) < low_loss_delta_threshold:
                print(
                    f"Loss change ({abs(prev_loss - loss)}) less than the threshold of {low_loss_delta_threshold}. Stopping optimization."
                )
                break

            prev_loss = loss

            if apply_rotation_angles_correction:
                with torch.no_grad():
                    self.params["pose"] = torch.atan2(
                        self.params["pose"].sin(), self.params["pose"].cos()
                    )

    def convert(
        self,
    ):

        for i, data in tqdm(enumerate(self.dataloader)):

            print(f"Processing batch {i}")

            if i == 0:
                self._init_params(
                    batch_size=self.cfg.data.batch_size, copy_from_prev=False
                )
            else:
                self._init_params(
                    batch_size=self.cfg.data.batch_size,
                    copy_from_prev=self.cfg.experiment.inherit_prev_batch_params_during_optimization,
                )

            target_vertices, faces = (
                data["vertices"].to(self.device),
                data["faces"],
            )  # .to(self.device)

            if self.cfg.experiment.edge_loss_optimization.use is True:
                print("Pose optimization using an edge loss")
                self._optimize(
                    target_vertices,
                    edge_loss,
                    [self.params["pose"]],
                    n_iters=self.cfg.experiment.edge_loss_optimization.n_iters,
                    regularization_weights=self.cfg.experiment.edge_loss_optimization.regularization_weights,
                    regularization_iters=self.cfg.experiment.edge_loss_optimization.regularization_iters,
                    low_loss_threshold=self.cfg.experiment.edge_loss_optimization.low_loss_threshold,
                    low_loss_delta_threshold=self.cfg.experiment.edge_loss_optimization.low_loss_delta_threshold,
                    apply_rotation_angles_correction=True,
                    loss_fn_args={"faces": faces},
                )

            if self.cfg.experiment.separate_global_translation_optimization.use is True:
                print("Global translation optimization using a vertex-to-vertex loss")
                self._optimize(
                    target_vertices,
                    vertex_loss,
                    [self.params["trans"]],
                    n_iters=self.cfg.experiment.separate_global_translation_optimization.n_iters,
                    regularization_weights=self.cfg.experiment.separate_global_translation_optimization.regularization_weights,
                    regularization_iters=self.cfg.experiment.separate_global_translation_optimization.regularization_iters,
                    low_loss_threshold=self.cfg.experiment.separate_global_translation_optimization.low_loss_threshold,
                    low_loss_delta_threshold=self.cfg.experiment.separate_global_translation_optimization.low_loss_delta_threshold,
                )

            print("Optimizing all parameters using a vertex-to-vertex loss")
            self._optimize(
                target_vertices,
                vertex_loss,
                [self.params["betas"], self.params["pose"], self.params["trans"]],
                n_iters=self.cfg.experiment.optimization.n_iters,
                regularization_weights=self.cfg.experiment.optimization.regularization_weights,
                regularization_iters=self.cfg.experiment.optimization.regularization_iters,
                low_loss_threshold=self.cfg.experiment.optimization.low_loss_threshold,
                low_loss_delta_threshold=self.cfg.experiment.optimization.low_loss_delta_threshold,
                apply_rotation_angles_correction=True,
            )

            with torch.no_grad():
                final_estimated_vertices = self.body_model(**self.params)["vertices"]

            print(
                f"Final v2v error: {v2v_error(final_estimated_vertices, target_vertices) * 1000} mm.\n"
            )

            self._save_results(
                file_name=f"batch_{i}",
                save_meshes=True,
                estimated_vertices=final_estimated_vertices,
                faces=faces,
            )

    def _save_results(
        self, file_name, save_meshes=False, estimated_vertices=None, faces=None
    ):

        with open(os.path.join(self.params_out_dir, file_name + ".pkl"), "wb") as f:
            pickle.dump(self.params, f)

        if save_meshes:
            assert estimated_vertices is not None
            assert faces is not None

            with open(os.path.join(self.meshes_out_dir, file_name + ".pkl"), "wb") as f:
                pickle.dump(
                    {
                        "vertices": estimated_vertices.detach().cpu().numpy(),
                        "faces": faces,
                    },
                    f,
                )
