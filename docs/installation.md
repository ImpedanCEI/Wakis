# ⚡ Installation guide

This section covers how to install `wakis` package for developers and users.

The installation guide is writen for Linux, but `wakis` code and dependencies are 100% python so it can run in other operating systems. For developers using Windows, we recommend checking the [WSL setup guide](https://learn.microsoft.com/en-us/windows/wsl/install) to install and setup the Windows subsystem for linux.

```{contents}
:depth: 3
```

## Installing from Github

### For users
`wakis` is now deployed to [PyPI](https://pypi.org/project/wakis/) and can be installed through pip:
```
pip install wakis
```
You can also upgrade to the latest version frequently by doing `pip install wakis --upgrade`.

To use `wakis` in python notebooks, the option `pip install wakis['notebook']` is preferred. See the section on [troubleshooting](#python-notebooks-troubleshooting) if an error pops up when rendering 3D plots.

### For developers
Wakis development is managed through [Wakis GitHub](https://github.com/ImpedanCEI/wakis). We encourage developers to use Github CLI from Windows/Linux. If you need to download it, refer to the official [Git Downloads](https://git-scm.com/downloads). To start using wakis and access the python scripts that compose the code, you can `git clone` it from the main repository:
```
# SSH:
git clone git@github.com:ImpedanCEI/wakis.git

# or HTTPS:
git clone https://github.com/ImpedanCEI/wakis.git
```

On a previously created conda environment, one can proceed installing the dependencies. On  `wakis` main directory do:
```
pip install -r requirements.txt
```

You can also pip install wakis on editable mode [RECOMMENDED]:
```
git clone git@github.com:ImpedanCEI/wakis.git
cd wakis
pip install -e .
```

If you would like to improve and push changes to `wakis`, we encorage to create a [fork](https://github.com/ImpedanCEI/wakis/fork) from wakis' `main` branch: https://github.com/ImpedanCEI/wakis on your personal GitHub.

To contribute, first fork the repository, create a new branch, and submit a pull request. Step-by-step:

1. Fork the repository: https://github.com/ImpedanCEI/wakis/fork
2. Create a new branch: `git checkout -b my-feature-branch`
3. Make your changes and commit them:
    `git add my-changed-script.py`
    `git commit -m 'Explain the changes'`
4. Push to the branch: git push origin my-feature-branch
5. Submit a pull request: https://github.com/ImpedanCEI/wakis/pulls

## Dependencies

`wakis` is a 100% pyhton code that relies only on a few renowed python packages:

* `numpy`: Used for numerical operations, especially for matrix operations.
* `scipy`: Provides additional functionality for sparse matrices and other scientific computations.
* `matplotlib`: Used for 1d and 2d plotting and visualization.
* `h5py`: This package provides a Python interface to the HDF5 binary data format for storing the field 3D data for several timesteps
* `tqdm`: This package is used for displaying progress bars in loops.
* `pyvista`: For handling and visualizing 3D CAD geometries and vtk-based 3D plotting.

To install only the dependencies in a conda python environment, simply run:

```
pip install -r requirements.txt
```

## Python installation

If a python installation has not been setup yet, we recommend using [miniforge](https://conda-forge.org/download/) (free of license) or [miniconda](https://docs.anaconda.com/free/miniconda/index.html) [^2].

Miniforge executable can be obtained from the website (Windows/Linux) [https://conda-forge.org/download/](https://conda-forge.org/download/), or from the terminal (Linux):

```
# get, install and activate miniforge
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
conda activate
```

Miniconda can also be installed and activated from the terminal:

```
# get, install and activate miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source miniconda3/bin/activate
```

We encourage the users to create a dedicated python environment with python <=3.11, >=3.9
```
# create dev python environment
conda create --name wakis-env python=3.11
conda activate wakis-env

# pip install wakis and other useful packages
pip install wakis                # minimal installation for scripts
pip install wakis['notebook']    # complete installation for notebook use
pip install bihc neffint iddefix # optional satellite packages
```

## GPU setup
`wakis` uses `cupy` to access GPU acceleration with the supported NVIDIA GPUs (more info in [Cupy's website](https://cupy.dev/)).
Provided a compatible [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-gpus) or [AMD ROCm GPUs](https://www.amd.com/en/products/graphics/desktops/radeon.html) with the adequate drivers & Toolkit, one can install `cupy` from pip:
```
# For CUDA 11.2 ~ 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# For AMD ROCm 4.3
pip install cupy-rocm-4-3

# For AMD ROCm 5.0
pip install cupy-rocm-5-0

# For other ROCm versions: See below
```
Or go for a `conda-forge` installation of both cupy and the CUDA Toolkit (RECOMMENDED):

```
conda install -c conda-forge cupy cuda-version=11.8
```

The toolkits can be installed using `conda`, `conda-forge`, `mamba` or from the source. Notice the version compatibility between GPU drivers and Toolkit to [choose the adequate version](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html):
```
# Conda alternative [RECOMENDED]
conda install cudatoolkit cuda-version=11 # for cuda 11.xx
conda install cuda-cudart cuda-version=12 # for cuda 12.xx

# Mamba alternative
conda install mamba -n base -c conda-forge
mamba install cudatoolkit=11.8.0
```

You can find the official CUDA Toolkit in the [NVIDIA development website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Fedora&target_version=41&target_type=rpm_local)

### CuPy setup on ROCm

The following configuration has been tested and confirmed to work as expected:
- ROCm 6.2.2
- Python 3.11
- CuPy 13.6.0 compiled from source on a HIP backend
- Ubuntu 22.04
- Radeon VII GPU (gfx906)

Compiling CuPy from source on ROCm can prove to be non-trivial, as such using different software/hardware configurations than the one described above can result in various compatibility issues.

On Debian distributions, the ROCm stack can be installed using:

```bash
sudo apt install rocm-hip-sdk rocm-dev rocm-opencl-runtime rocm-smi-lib
```

The above path will typically install the latest available version. To install a specific (or older) version, add the AMD ROCm repo. For ROCm-6.2.2, this involves the following steps:
- Add AMD driver key:
    ```bash
    wget https://repo.radeon.com/rocm/rocm.gpg.key
    sudo mv rocm.gpg.key /usr/share/keyrings/rocm.gpg
    ```
- Add ROCm 6.2.2
    ```bash
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.2 jammy main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt update
    ```
- Then install using the usual command

Once the installation has been complete or if the stack has been previously installed on the system, the stack should be available under one of the following paths:

```bash
/opt/rocm # often a symlink to a specific version
/opt/rocm-v.v.v/ # Where v.v.v represents a specific version number
```

There should be only *one* ROCm version present. If multiple versions can be found under `/opt`, it is recommended to clean install the driver. Once the driver path has been identified, we must identify the GPU name, which can be found using:

```bash
rocminfo | grep Name
```

This should yield an output resembling the following:
```
Name:                    gfx906    <- This is the desired name
Marketing Name:          AMD Radeon VII
Vendor Name:             AMD
    Name:                    amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-
```

The GPU architecture name should be of the form `gfx...`. Once these parameters have been specified, cupy can be installed inside the python environment by setting the following environment variables and installing from source:

```bash
export ROCM_HOME=/opt/rocm # Generally safe, but check paths
export HIPCC="$ROCM_HOME/bin/hipcc"
export CXX="$HIPCC"
export PATH="$ROCM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_HOME/lib:$ROCM_HOME/lib64:${LD_LIBRARY_PATH}"

export HCC_AMDGPU_TARGET=gfx906 # Replace with your GPU architecture
export CUPY_INSTALL_USE_HIP=1

python -m pip install --no-cache-dir --force-reinstall "cupy==13.6.0"

```
Depending on driver versions, the paths shown above can change. If the cupy installation fails, it is recommended to verify that these paths are valid on your system.

### HTCondor with GPU submission

Your `config_condor.sub` file needs to request a GPU
```
if !defined FNAME
    FNAME               = outputs
endif

ID                      = $(Cluster).$(Process)

output                  = ./logs/$(FNAME).$(ID).out
error                   = ./logs/$(FNAME).$(ID).err
log                     = ./logs/$(FNAME).$(Cluster).log

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

request_GPUs            = 1
request_CPUs            = 1
+MaxRunTime             = 100
+AccountingGroup        = your_acct_group

executable              = simulation_run.sh
```

Your `simulation_run.sh` file could optionally include:

```
#!/bin/bash
# source the miniforge
source /afs/cern.ch/work/u/user/miniforge3/etc/profile.d/conda.sh
conda activate /afs/cern.ch/work/u/user/SimulationProjects/wakis_simulation/wakis-env

# Fix for UCX errors
export UCX_TLS=rc,tcp,cuda_copy,cuda_ipc

# Fix for HDF5 file locking issues, when overwriting to same directory
export HDF5_USE_FILE_LOCKING=FALSE

python3.11 -m wakis_simulation.main
```

An example python project for simulating on HTCondor, is shown here [BLonD Simulation Template](https://gitlab.cern.ch/blond/blond-simulation-template).
The submission files are based on the approach used in this example project [^3], modified for GPU. The project can be modified for the wakis environment, it currently is set up
for simulations with BLonD, the longitudinal beam dynamics code.

## CPU Multithreading
To achieve multithreading for Wakis `sparse matrix x vector` operations, the Intel-MKL backend has been implemented as an alternative to single-threaded `scipy.sparse.dot`. To install it in your conda environment simply do:

```
conda create --name wakis-env python=3.12 numpy scipy mkl mkl-service
pip install sparse_dot_mkl
pip install wakis['notebook']
```

Wakis will detect that the package is installed and use it as default backend. To control the number of threads and memory pinning, one can add the following lines to your python script **before** the imports:

```python
# [!] Set before importing numpy/scipy/mkl modules
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())       # Number of OpenMP threads
os.environ["KMP_AFFINITY"] = "balanced,granularity=fine"  # Thread pinning
```

* 🔜 A domain decomposition multithreading is envisioned for future resleases

## CPU/GPU MPI setup
To run multi-node CPU parallelized simulations, Wakis needs the following packages:

* OpenMPI installed in your operating system:
* Python package [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/)
* Python package [`ipyparallel`](https://ipyparallel.readthedocs.io/en/latest/tutorial/intro.html) to start a MPI kernel inside notebooks

The preferred install method is through `conda-forge`:

```bash
# All at once [recommended]
conda install -c conda-forge mpi4py openmpi
```

In case openmpi fails, sometimes installing them separately helps:

```bash
# Conda forge openmpi
conda install -c conda-forge openmpi

# Check install
which mpicc #/usr/bin/mpicc
which mpiexec #/usr/bin/mpiexec

# Conda forge mpi4py
conda install -c conda-forge mpi4py

# For working on jupyter notebooks:
pip install ipyparallel
```

Alternatively, for sudo users, installation through `apt`+`pip` has also proven to work:
```bash
# sudo alternative (ubuntu)
sudo apt update
sudo sudo apt install openmpi-bin openmpi-common libopenmpi-dev
# mpi4py through pip
CC=mpicc pip install --no-cache-dir mpi4py
# For working on jupyter notebooks:
pip install ipyparallel
```
### Checking multi-GPU compatibility

* `OpenMPI` is built in Linux with multi-GPU compatibility (provided `cupy` is correctly setup):
```bash
# To check CUDA-aware ompi, try:
ompi_info --parsable | grep mpi_built_with_cuda_support
# output should be:
# mca:mpi:base:param:mpi_built_with_cuda_support:value:true

# Alternatively, try:
ompi_info | grep -i cuda
# output should contain:
# MPI extensions: affinity, cuda, pcollreq
```
Before launching an MPI simulation, make sure to set the environment variable `OMPI_MCA_opal_cuda_support` to `true`:

```bash
# Set environment variable to use CUDA-aware before launching your MPI processes
export OMPI_MCA_opal_cuda_support=true
```
Or from python, before importing `mpi4py`:
```python
import os
os.environ['OMPI_MCA_opal_cuda_support']='true'
```

### Running multi-GPU simulation

To run MPI simulations from the terminal:
```bash
# multi CPU
mpiexec -n 8 python your_wakis_cpu_script.py

# multi GPU
mpiexec --mca btl_smcuda_cuda_ipc_max 0 -n 4 python your_wakis_gpu_script.py
# or:
mpiexec --mca opal_cuda_support 1 -n 4 python your_wakis_gpu_script.py
```

It is also possible to run multi-GPU from **python notebooks** using `ipyparallel` and the notebook magic `%%px`:
```python
import ipyparallel as ipp
cluster = ipp.Cluster(engines="mpi", n=2).start_and_connect_sync()
```
Once the cluster has initialized:
```python
%%px
import os
os.environ['OMPI_MCA_opal_cuda_support']='true' # multi GPU

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Process {rank} of {size} is running")

import cupy
cupy.cuda.Device(rank).use()
```

## Python Notebooks troubleshooting

### Matplotlib interactive plots
Within jupyter notebooks, in order to be able to zoom and interact with matplotlib figures, one needs to use notebook magic commands `%`.
* The recommended one for Jupyter notebooks on the web is `%matplotlib widget`
* The recommended one for Jupyter notebooks on VS Code in `%matplotlib ipympl`

The package `ipympl`can be easily installed using `pip install ipympl`

### PyVista interactive plots
To be able to render 3D interactive plots in Jupyter notebooks, it is recommended to use the `wakis['notebook']` pip installation.

Some driver problems may arise depending on pre-installed versions. One way of solving common errors like `MESA-loader` or `libGL error` is installing a new driver within your conda environment with:

```
conda install conda-forge::libstdcxx-ng
```

If you're in a headless environment (e.g., remote server, openstack machine), forcing OSMesa (Off-Screen Mesa) rendering might help:

```python
import os
os.environ['PYVISTA_USE_OSMESA'] = 'True'
```

----

[PyVista](https://github.com/pyvista/pyvista) a python package for 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK) [^1].
We rely on [PyVista](https://github.com/pyvista/pyvista) to import CAD/STL geometry as embedded boundaries or/and as solids with different materials into the domain. This allows `wakis` to efficiently render 3D plots of the electromagnetic fields and visualize the computational mesh interactively.



[^1]: Sullivan and Kaszynski, (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). Journal of Open Source Software, 4(37), 1450, https://doi.org/10.21105/joss.01450
[^2]: Anaconda Software Distribution. (2020). Anaconda Documentation. Anaconda Inc. Retrieved from https://docs.anaconda.com/
[^3]: S. Lauber, (2025). Blond Simulation Template. Retrieved from https://gitlab.cern.ch/blond/blond-simulation-template
