# uncertainty-buffer

Here we introduce a new sampling strategy for buffers in reinforcement learning that is based on critic estimates.

## Environment Setup for Conda

The experiments rely on MuJoCo being installed. Follow the instructions from [`mujoco-py`](https://github.com/openai/mujoco-py#install-mujoco) and download the archive with MuJoCo (version 2.1.0). Then extract the file into the standard path `~/.mujoco/mujoco210`.

After this step, we use [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) to create a `python` environment with all the dependencies:
```bash
conda env create -f ./scripts/environment.yml
conda activate mujoco
```

This will install the latest versions of the packages in `./scripts/environment.yml`. If you want to use exactly the same versions we use (and you are on on a `linux-64` platform) you can use

```bash
conda create --name mujoco --file ./scripts/explicit_env_specs_linux64.txt
conda activate mujoco
```

In order to run the codes you then need to activate the `conda` environment and set some variables (to allow MuJoCo to compile some libraries).
We used `gcc` v8.3.0 and added the following `env` variables:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<YOUR-HOME>/.mujoco/mujoco210/bin
export CPATH=<YOUR-CONDA-PREFIX>/envs/mujoco/include
```

where you should change `<YOUR-HOME>` to your home folder and `<YOUR-CONDA-PREFIX>` to the loation of the `conda` installation.

The code is tested with

* Python (3.9.12)
* Python (3.10.4)
## Code execution from terminal

### arguments

- ```-- buffer (Optional, default:uniform)``` : uncertainty (MEET), prioritized or uniform

- ```-- buffer_size (Optional, default:1e5)``` : buffer capacity in the format of 1e5

- ```-- alg (Optional, default:sac)``` : implemented algorithms are sac, ddpg, td3

- ```-- n_epochs (Optional, default:1000)``` : number of epochs. Steps per epoch is fixed to 1000. Thus, n_epochs=1000 result in 1 million steps.

- ```-- n_experiments (Optional, default:1)``` : number of experiments for sequential execution

- ```-- env (Optional, default:Humanoid-v3)``` : Choose a desired mujoco environment out of 
    - Humanoid-v3
    - Ant-v3,
    - HalfCheetah-v3
    - Walker2d-v3
    - InvertedPendulum-v2
    - InvertedDoublePendulum-v2
    - HumanoidStandup-v2
    - Reacher-v2
    - Swimmer-v3
    - Hopper-v3


### Example command
```python
python playground/reinforcement_learning_multi_head.py --buffer=uncertainty --buffer_size=1e5 --alg=sac --n_epochs=1000 --n_experiments=1 --env=Walker2d-v3

```

## Google Colab Execution
### These steps need no further installation and works out of the box

1. upload example_notebook.ipynb in google colab
2. compress the the directory with the code to a zip file
3. follow the steps in the explanation video
