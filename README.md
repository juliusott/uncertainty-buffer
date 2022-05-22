# uncertainty-buffer
Introduce uncertainty sampling based on critic estimates

## Environment Setup for Conda

```bash
conda env create -f ./scripts/environment.yml
```

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
python playground/reinforcement_learning_multi_head.py --buffer=uncertainty --buffer_size=1e6 --alg=sac --n_epochs=1000 --n_experiments=2 --env=Walker2d-v3

```

