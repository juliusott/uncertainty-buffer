from re import A
import numpy as np
import argparse
import sys
sys.path.append('/home/sanchez/mda_mrtl/nodm/mrtl/units/metartl/home/mujoco/uncertainty-buffer/')
print(sys.path)
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
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange
import os

from networks.networks import MultiHeadCriticNetwork, ActorNetwork, MultiHeadCriticNetwork_noise, SACActorNetwork


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
    initial_replay_size = 300
    
    max_replay_size = 100000
    batch_size = 256
    n_features = 256
    tau = .001
    warmup_transitions = 1000
    use_noise = False
    if use_noise:
        critic_network = MultiHeadCriticNetwork_noise
    else:
        critic_network = MultiHeadCriticNetwork

    if "SAC" in alg.__name__ or "sac" in alg.__name__:
        print("loading sac actor network")
        actor_network = SACActorNetwork
    else:
        actor_network = ActorNetwork

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=actor_network,
                        n_features=n_features,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        use_cuda=use_cuda)

    actor_sigma_params = dict(network=actor_network,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)
    
    if "SAC" in alg.__name__ or "sac" in alg.__name__ :
        actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': 3e-4}}
    else:
        actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': .001}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=critic_network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         head_prob = 0.7,
                         use_cuda=use_cuda)

    # Agent
    if "SAC" in alg.__name__ or "sac" in alg.__name__ :
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
    s = None
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=3)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        s, *_ = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, gamma))
        R = np.mean(compute_J(dataset))
        if "SAC" in alg.__name__:
            E = agent.policy.entropy(s)
        else:
            E = None
        rewards.append(R)
        if n % 100 == 0 :
            #agent.save_buffer_snapshot(alg_name=alg.__name__,  epoch=n+1)
            np.save(filename, np.array(rewards))
        logger.epoch_info(n+1, J=J, R=R, entropy=E)
    np.save(filename, np.array(rewards))
    #logger.info('Press a button to visualize pendulum')
    #input()
    #core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='define experimental setup')
    parser.add_argument('--buffer', help='buffer sampling strategy [uniform, uncertainty, prioritized]', type=str, default="uniform")
    parser.add_argument('--alg', help='choose algorithm [SAC, DDPG, TD3]', type=str, default="sac")
    parser.add_argument('--n_epochs', help='number of epochs per iteration', type=int, default=1000)
    parser.add_argument('--n_experiments', help='number of experiments ', type=int, default=1)
    args = parser.parse_args()
    
    buffer_strategy = args.buffer
    algorithm = args.alg
    n_epochs = int(args.n_epochs)
    n_experiments = int(args.n_experiments)
    if algorithm == "sac" or algorithm.lower() == "sac":
        alg = MultiHeadSAC
    elif algorithm == "td3":
        alg = MultiHeadTD3
    else:
        alg = MultiHeadDDPG

    for _ in range(n_experiments):
        experiment(alg=alg, n_epochs=n_epochs, n_steps=1000, n_steps_test=2000, buffer_strategy=buffer_strategy, buffer_alpha=1, buffer_beta=1)
