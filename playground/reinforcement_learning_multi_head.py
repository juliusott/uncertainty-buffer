from re import A
import numpy as np
import argparse

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
import os

from networks.networks import MultiHeadCriticNetwork, ActorNetwork, MultiHeadCriticNetwork_noise


def experiment(alg, n_epochs, n_steps, n_steps_test, buffer_strategy, buffer_alpha, buffer_beta):
    #np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    use_cuda = torch.cuda.is_available()

    # MDP
    horizon = 100
    gamma = 0.99
    env_name = 'Humanoid-v3'
    mdp = Gym(env_name, horizon, gamma)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

    # Settings
    initial_replay_size = 550
    max_replay_size = int(2**18)
    batch_size = 256
    n_features = 64
    tau = .001
    warmup_transitions = 100
    use_noise = True
    if use_noise:
        critic_network = MultiHeadCriticNetwork_noise
    else:
        critic_network = MultiHeadCriticNetwork

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
    critic_params = dict(network=critic_network,
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
        tau = 0.005
        lr_alpha = 3e-4
        agent = alg(mdp.info, actor_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, buffer_strategy=buffer_strategy, buffer_alpha=buffer_alpha, buffer_beta=buffer_beta)
    else:
        agent = alg(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau, warmup_transitions=warmup_transitions, 
                buffer_strategy=buffer_strategy)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = np.mean(compute_J(dataset, gamma))
    R = np.mean(compute_J(dataset))

    logger.epoch_info(0, J=J, R=R)
    rewards = list()
    k= 1
    filename = f"{alg.__name__}{k}_{env_name}_{buffer_strategy}_noise{use_noise}_alpha{buffer_alpha}_beta{buffer_beta}.npy"
    while os.path.isfile(filename):
        filename = f"{alg.__name__}{k}_{env_name}_{buffer_strategy}_noise{use_noise}_alpha{buffer_alpha}_beta{buffer_beta}.npy"
        k +=1 
    print(f"save file {filename}")
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = np.mean(compute_J(dataset, gamma))
        R = np.mean(compute_J(dataset))
        rewards.append(R)
        if n % 100 == 0 :
            #agent.save_buffer_snapshot(alg_name=alg.__name__,  epoch=n+1)
            np.save(filename, np.array(rewards))
        logger.epoch_info(n+1, J=J, R=R)
    np.save(filename, np.array(rewards))
    #logger.info('Press a button to visualize pendulum')
    #input()
    #core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='define experimental setup')
    parser.add_argument('--buffer', help='buffer sampling strategy [uniform, uncertainty, prioritized]', type=str, default="uniform")
    parser.add_argument('--alg', help='choose algorithm [SAC, DDPG, TD3]', type=str, default="SAC")
    parser.add_argument('--n_epochs', help='number of epochs per iteration', type=int, default=1000)
    parser.add_argument('--n_experiments', help='number of experiments ', type=int, default=1)
    args = parser.parse_args()
    
    buffer_strategy = args.buffer
    algorithm = args.alg
    n_epochs = int(args.n_epochs)
    n_experiments = int(args.n_experiments)
    if algorithm == "SAC":
        alg = MultiHeadSAC
    elif algorithm == "TD3":
        alg = MultiHeadTD3
    else:
        alg = MultiHeadDDPG

    for _ in range(n_experiments):
        for buffer_alpha in [0.1, 0.2, 0.4, 0.6, 0.8, 1]:
            for buffer_beta in [0.9, 0.7, 0.5, 0.3, 0.1]:
                experiment(alg=alg, n_epochs=n_epochs, n_steps=1000, n_steps_test=1000, buffer_strategy=buffer_strategy, buffer_alpha=buffer_alpha, buffer_beta=buffer_beta)
