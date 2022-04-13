import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.multi_head_ddpg import MultiHeadDDPG
from agents.multi_head_TD3 import MultiHeadTD3
from agents.mult_head_sac import MultiHeadSAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.dataset import compute_J

from tqdm import trange

from networks.networks import MultiHeadCriticNetwork, ActorNetwork


def experiment(alg, n_epochs, n_steps, n_steps_test):
    #np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    use_cuda = torch.cuda.is_available()

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gym('Humanoid-v3', horizon, gamma)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

    # Settings
    initial_replay_size = 500
    max_replay_size = 5000
    batch_size = 200
    n_features = 64
    tau = .001

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        use_cuda=use_cuda)

    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': .001}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=MultiHeadCriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': .001}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         head_prob = 0.7,
                         use_cuda=use_cuda)

    # Agent
    if "SAC" in alg.__name__:
        warmup_transitions = 100
        tau = 0.005
        lr_alpha = 3e-4
        agent = alg(mdp.info, actor_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)
    else:
        agent = alg(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = np.mean(compute_J(dataset, gamma))
    R = np.mean(compute_J(dataset))

    logger.epoch_info(0, J=J, R=R)
    rewards = list()
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = np.mean(compute_J(dataset, gamma))
        R = np.mean(compute_J(dataset))
        rewards.append(R)
        logger.epoch_info(n+1, J=J, R=R)

    np.save( alg.__name__+".npy", np.asarray(rewards))
    #logger.info('Press a button to visualize pendulum')
    #input()
    #core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    algs = [MultiHeadTD3, MultiHeadDDPG, MultiHeadSAC]

    for alg in algs:
        experiment(alg=alg, n_epochs=4000, n_steps=1000, n_steps_test=2000)
