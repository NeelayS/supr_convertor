import os

import numpy as np
import torch
import trimesh
from smplx.body_models import create


def generate_smplx_meshes(params_path, model_path, output_dir, output_format="ply"):

    os.makedirs(output_dir, exist_ok=True)

    params = np.load(params_path, allow_pickle=True)

    trans = torch.tensor(params["trans"]).float()

    betas = torch.tensor(params["betas"]).float()
    n_betas = len(betas)
    betas = betas.unsqueeze(0)

    model = create(
        model_path,
        model_type="smplx",
        gender=params["gender"],
        num_betas=n_betas,
        use_pca=False,
    )

    poses = torch.tensor(params["poses"]).float()
    n_bodies = poses.shape[0]

    global_orient = poses[:, :3]
    body_pose = poses[:, 3:66]
    jaw_pose = poses[:, 66:69]
    left_eye_pose = poses[:, 69:72]
    right_eye_pose = poses[:, 72:75]
    left_hand_pose = poses[:, 75:120]
    right_hand_pose = poses[:, 120:165]

    if "expression" in params.keys():
        expression = torch.tensor(params["expression"]).float()
    else:
        expression = torch.zeros(1, 10)

    for pose_idx in range(n_bodies):
        pose_idx = [pose_idx]

        output = model(
            betas=betas,
            transl=trans[pose_idx],
            global_orient=global_orient[pose_idx],
            body_pose=body_pose[pose_idx],
            left_hand_pose=left_hand_pose[pose_idx],
            right_hand_pose=right_hand_pose[pose_idx],
            jaw_pose=jaw_pose[pose_idx],
            leye_pose=left_eye_pose[pose_idx],
            reye_pose=right_eye_pose[pose_idx],
            expression=expression,
            return_verts=True,
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        mesh = trimesh.Trimesh(vertices, model.faces, process=False)

        output_path = os.path.join(
            output_dir, str(pose_idx[0]).zfill(6) + "." + output_format
        )
        mesh.export(output_path)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_format", type=str, default="ply")

    args = parser.parse_args()

    generate_smplx_meshes(
        args.params_path, args.model_path, args.output_dir, args.output_format
    )
