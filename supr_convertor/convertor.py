import os
import pickle
from typing import Callable, Union

import torch
import trimesh
from tqdm import tqdm

from supr_convertor.config import CfgNode
from supr_convertor.data import MeshFolderDataset
from supr_convertor.losses import edge_loss, v2v_error, vertex_loss
from supr_convertor.utils import seed_everything


class Convertor:
    """
    Convertor class to fit body model to meshes
    """

    def __init__(
        self,
        cfg: CfgNode,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader = None,
        mesh_dir: str = None,
        out_dir: str = None,
        device: torch.device = None,
    ):

        self.cfg = cfg
        self.model = model
        self.device = device

        if dataloader is None:
            assert mesh_dir is not None and os.path.exists(mesh_dir)

            self.dataloader = torch.utils.data.DataLoader(
                MeshFolderDataset(mesh_dir),
                batch_size=self.cfg.data.batch_size,
                shuffle=False,
                num_workers=1,
            )
        else:
            self.dataloader = dataloader

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
                self.device = torch.device(self.cfg.device)
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

        self.model.to(self.device)

        self.betas = None
        self.pose = None
        self.trans = None

    def _init_params(
        self,
        batch_size: int,
        copy_from_prev: bool = False,
        betas_without_grad: bool = False,
    ):
        """
        Initializes the parameters of the body model

        Parameters
        ----------
        batch_size: int
            batch size
        copy_from_prev: bool
            Whether to initialize with the values of the previous batch
        betas_without_grad: bool
            Whether to optimize the values of the betas. If False, the values of the betas are fixed
        """

        if copy_from_prev is True:
            assert (
                self.betas is not None
                and self.pose is not None
                and self.trans is not None
            ), "Parameters must be initialized from scratch for the first batch"

            self.betas = self.betas.clone().detach().requires_grad_(True)
            self.pose = self.pose.clone().detach().requires_grad_(True)
            self.trans = self.trans.clone().detach().requires_grad_(True)

        else:
            self.betas = torch.zeros(
                (batch_size, self.cfg.model.n_betas),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.pose = torch.zeros(
                (batch_size, 225),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.trans = torch.zeros(
                (batch_size, 3),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )

        if betas_without_grad is True:
            assert (
                self.betas is not None
            ), "Betas must be optimized at least once for a sequence"

            self.betas = self.betas.clone().detach().requires_grad_(False)
            self.betas = torch.mean(self.betas, dim=0, keepdim=True)
            self.betas = self.betas.repeat(batch_size, 1)

    def _optimize(
        self,
        target_vertices: torch.Tensor,
        loss_fn: Callable,
        params_to_optimize: list,
        n_iters: int,
        loss_fn_kwargs: dict = {},
        apply_rotation_angles_correction: bool = False,
        low_loss_threshold: float = 2e-3,
        low_loss_delta_threshold: float = 1e-6,
        n_consecutive_low_loss_delta_iters_threshold: int = 5,
        gradient_clip: float = None,
        params_regularization_weights: Union[tuple, list] = None,
        params_regularization_iters: Union[tuple, list] = None,
    ):

        if params_regularization_weights is not None:
            assert (
                params_regularization_iters is not None
            ), "params_regularization_iters must be provided if params_regularization_weights is provided"
            assert len(params_regularization_weights) == len(
                params_regularization_iters
            ), "params_regularization_weights and params_regularization_iters should have the same number of values"

        optimizer = torch.optim.LBFGS(
            params_to_optimize, **self.cfg.experiment.optimizer_params.to_dict()
        )

        prev_loss = 1e10
        n_consecutive_low_loss_delta_iters = 0
        low_loss_delta_hit_iter_idx = 0
        last_k_losses = []

        for n_iter in tqdm(range(n_iters), desc="Optimizing..."):

            def closure():

                optimizer.zero_grad()

                estimated_vertices = self.model(
                    betas=self.betas, pose=self.pose, trans=self.trans
                )["vertices"]
                loss = loss_fn(estimated_vertices, target_vertices, **loss_fn_kwargs)

                if params_regularization_weights is not None:
                    for i, (weight, regularization_iters) in enumerate(
                        zip(params_regularization_weights, params_regularization_iters)
                    ):
                        if n_iter < regularization_iters:
                            loss += weight * torch.mean(params_to_optimize[i] ** 2)

                loss.backward()

                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, gradient_clip)

                return loss

            loss = optimizer.step(closure)

            if apply_rotation_angles_correction is True:
                with torch.no_grad():
                    self.pose[:] = torch.atan2(self.pose.sin(), self.pose.cos())

            if n_iter % self.cfg.log_iterations_interval == 0:
                print(f"Iteration {n_iter + 1}/{n_iters} | Loss: {loss.item():.8f}")

            if loss.item() < low_loss_threshold:
                print(
                    f"Loss threshold ({low_loss_threshold}) reached at iteration {n_iter + 1}. Stopping optimization."
                )
                print(
                    f"Last {n_consecutive_low_loss_delta_iters_threshold} losses: {last_k_losses[1:]}"
                )
                break

            if abs(loss.item() - prev_loss) < low_loss_delta_threshold:
                if n_consecutive_low_loss_delta_iters == 0:
                    n_consecutive_low_loss_delta_iters += 1
                else:
                    if n_iter - low_loss_delta_hit_iter_idx == 1:
                        n_consecutive_low_loss_delta_iters += 1
                    else:
                        n_consecutive_low_loss_delta_iters = 1

                low_loss_delta_hit_iter_idx = n_iter

            if n_iter >= n_consecutive_low_loss_delta_iters_threshold + 1:
                last_k_losses.pop(0)
            last_k_losses.append(loss.item())

            if (
                n_consecutive_low_loss_delta_iters
                >= n_consecutive_low_loss_delta_iters_threshold
            ):
                print(
                    f"Low loss delta threshold ({low_loss_delta_threshold}) for {n_consecutive_low_loss_delta_iters_threshold} consecutive iterations reached at iteration {n_iter + 1}. Stopping optimization."
                )
                print(
                    f"Last {n_consecutive_low_loss_delta_iters_threshold} losses: {last_k_losses[1:]}"
                )
                print(
                    f"Last {n_consecutive_low_loss_delta_iters_threshold} loss deltas: {[last_k_losses[i] - last_k_losses[i+1] for i in range(n_consecutive_low_loss_delta_iters_threshold)]}"
                )
                print(f"Final loss: {loss.item()}")
                break

            prev_loss = loss.item()

    def convert(
        self,
    ):

        last_batch_size = len(self.dataloader.dataset) % self.cfg.data.batch_size
        if last_batch_size == 0:
            last_batch_size = self.cfg.data.batch_size

        for n_batch, data in enumerate(self.dataloader):

            if n_batch == len(self.dataloader) - 1:
                print(
                    f"Last batch in dataloader. Using a batch size of {last_batch_size}"
                )
                batch_size = last_batch_size
            else:
                batch_size = self.cfg.data.batch_size

            if n_batch == 0:
                self._init_params(batch_size=batch_size)

            else:
                if self.cfg.experiment.optimize_betas_only_for_first_batch is True:
                    self._init_params(batch_size=batch_size, betas_without_grad=True)
                else:
                    self._init_params(
                        batch_size=batch_size,
                        copy_from_prev=self.cfg.experiment.inherit_prev_batch_params_during_optimization,
                    )

            print(f"\nProcessing batch {n_batch + 1}/{len(self.dataloader)}\n")

            target_vertices = data["vertices"].to(self.device)
            faces = data["faces"]
            if faces.ndim == 3:
                faces = faces[0]

            if self.cfg.experiment.edge_loss_optimization.use is True:
                print("\nPerforming pose optimization using an edge loss\n")
                self._optimize(
                    target_vertices=target_vertices,
                    loss_fn=edge_loss,
                    params_to_optimize=[self.pose],
                    n_iters=self.cfg.experiment.edge_loss_optimization.n_iters,
                    loss_fn_kwargs={"faces": faces},
                    apply_rotation_angles_correction=self.cfg.experiment.edge_loss_optimization.apply_rotation_angles_correction,
                    low_loss_threshold=self.cfg.experiment.edge_loss_optimization.low_loss_threshold,
                    low_loss_delta_threshold=self.cfg.experiment.edge_loss_optimization.low_loss_delta_threshold,
                    n_consecutive_low_loss_delta_iters_threshold=self.cfg.experiment.edge_loss_optimization.n_consecutive_low_loss_delta_iters_threshold,
                    gradient_clip=self.cfg.experiment.edge_loss_optimization.gradient_clip,
                    params_regularization_weights=self.cfg.experiment.edge_loss_optimization.params_regularization_weights,
                    params_regularization_iters=self.cfg.experiment.edge_loss_optimization.params_regularization_iters,
                )

            if self.cfg.experiment.vertex_to_vertex_loss_type == "v2v_error":
                vertex_to_vertex_loss = v2v_error
            else:
                vertex_to_vertex_loss = vertex_loss

            if self.cfg.experiment.separate_global_translation_optimization.use is True:
                print(
                    "\nPerforming global translation optimization using a vertex-to-vertex loss\n"
                )
                self._optimize(
                    target_vertices=target_vertices,
                    loss_fn=vertex_to_vertex_loss,
                    params_to_optimize=[self.trans],
                    n_iters=self.cfg.experiment.separate_global_translation_optimization.n_iters,
                    loss_fn_kwargs={
                        "reduction": self.cfg.experiment.separate_global_translation_optimization.vertex_loss_reduction
                    },
                    apply_rotation_angles_correction=False,
                    low_loss_threshold=self.cfg.experiment.separate_global_translation_optimization.low_loss_threshold,
                    low_loss_delta_threshold=self.cfg.experiment.separate_global_translation_optimization.low_loss_delta_threshold,
                    n_consecutive_low_loss_delta_iters_threshold=self.cfg.experiment.separate_global_translation_optimization.n_consecutive_low_loss_delta_iters_threshold,
                    gradient_clip=self.cfg.experiment.separate_global_translation_optimization.gradient_clip,
                )

            print("\nOptimizing all parameters using a vertex-to-vertex loss\n")
            self._optimize(
                target_vertices=target_vertices,
                loss_fn=vertex_to_vertex_loss,
                params_to_optimize=[self.betas, self.pose, self.trans],
                n_iters=self.cfg.experiment.optimization.n_iters,
                loss_fn_kwargs={
                    "reduction": self.cfg.experiment.optimization.vertex_loss_reduction
                },
                apply_rotation_angles_correction=self.cfg.experiment.optimization.apply_rotation_angles_correction,
                low_loss_threshold=self.cfg.experiment.optimization.low_loss_threshold,
                low_loss_delta_threshold=self.cfg.experiment.optimization.low_loss_delta_threshold,
                n_consecutive_low_loss_delta_iters_threshold=self.cfg.experiment.optimization.n_consecutive_low_loss_delta_iters_threshold,
                gradient_clip=self.cfg.experiment.optimization.gradient_clip,
                params_regularization_weights=self.cfg.experiment.optimization.params_regularization_weights,
                params_regularization_iters=self.cfg.experiment.optimization.params_regularization_iters,
            )

            final_estimated_vertices = self.model(
                betas=self.betas, pose=self.pose, trans=self.trans
            )["vertices"]
            final_v2v_error = v2v_error(final_estimated_vertices, target_vertices)

            print(
                f"\nOptimization complete. Final v2v error: {final_v2v_error * 1000} mm"
            )
            print("Saving results")

            final_estimated_vertices = final_estimated_vertices.detach().cpu().numpy()

            with open(
                os.path.join(
                    self.params_out_dir, "batch_" + str(n_batch).zfill(6) + ".pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "betas": self.betas.detach().cpu().numpy(),
                        "pose": self.pose.detach().cpu().numpy(),
                        "trans": self.trans.detach().cpu().numpy(),
                    },
                    f,
                )

            if self.cfg.save_output_meshes is True:
                for sample in range(batch_size):
                    idx = n_batch * self.cfg.data.batch_size + sample

                    output_mesh = trimesh.Trimesh(
                        vertices=final_estimated_vertices[sample],
                        faces=faces,
                        process=False,
                    )
                    output_mesh.export(
                        str(
                            os.path.join(
                                self.meshes_out_dir, str(idx).zfill(6) + ".ply"
                            )
                        )
                    )
