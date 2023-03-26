# SUPR Convertor

A simple tool to convert [SMPL-X](https://smpl-x.is.tue.mpg.de/) model parameters to [SUPR](https://supr.is.tue.mpg.de/) model parameters.

## Installation

1. Clone the repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/NeelayS/supr_convertor.git
```

2. Install the PyTorch version of your choice, which is compatible with your GPU, from the [official website](https://pytorch.org/).

3. Install the dependencies for this repository:

```bash
python setup.py install
cd SUPR && python setup.py install && cd ..
```

## Usage

1. Download the SUPR model(s) from the official [project website](https://supr.is.tue.mpg.de/).

2. Use the `generate_smplx_meshes.py` to script to generate meshes from SMPL-X parameters and save them as either `.ply` or `.obj` files.

First, install the [SMPL-X](https://github.com/vchoutas/smplx) package. <br>

Then, run the script as follows:

```bash
python generate_smplx_meshes.py --params_path <path_to_smplx_params> --model_path <path_to_smplx_model> --output_dir <path_to_output_dir> --output_format <ply/obj>
```
The `smplx_params_path` should be a `.npz` file containing the SMPL-X parameters. The `.npz` file should contain the different model parameters and metadata such as <b>gender</b>. An example file has been provided in this repository under `data`. <br>
The `output_dir` should be a directory where the generated meshes will be saved. The `output_format` can be either `obj` or `ply`.

If you don't have the SMPL-X model file(s), you can download them from the [official website](https://smpl-x.is.tue.mpg.de/). <br>

<b> Note: </b> You can skip this step if you already have SMPL-X meshes in `.obj` or `.ply` format.

3. Modify the base config file `configs/base_config.yaml` to suit your needs. In particular, you must specify the following parameters:

```yaml
data:
    mesh_dir: <path_to_smplx_meshes>

model:
    gender: <gender_of_supr_body_model>
    path: <path_to_supr_model>

device: <device_to_use>
out_dir: <path_to_output_dir>
```
The rest of the parameters can be left as is, or modified as per your requirements. <br> 
To convert the parameters in the least amount of time, set the `batch_size` to the maximum possible value, constrained by the available device memory. <br>
The number of iterations of the optimization process for the conversion and the stopping conditions can be modified according to the use case. <br>

<b> Note: </b>: Be careful to use the same gendered SUPR model as the SMPL-X model used to generate the meshes.

4. Run the `convert.py` script to convert the parameters.

```bash
python convert.py --cfg <path_to_your_config_file>
```

The converted SUPR parameters will be saved in the `out_dir` specified in the config file. Optionally, SUPR meshes obtained from the converted parameters can be saved as well.
