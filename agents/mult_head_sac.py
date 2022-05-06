import numpy as np

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from approximators.masked_torch_regressor import  MaskedTorchApproximator
from buffer.uncertainty_buffer import UncertaintyReplayMemory
from buffer.prioritized_buffer import PrioritizedReplayMemory
from mushroom_rl.utils.replay_memory import ReplayMemory #, PrioritizedReplayMemory
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import chain
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.sac import SACPolicy


class MultiHeadSAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.
    """
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions= 100, tau=0.005,
                 lr_alpha=3e-4, log_std_min=-20, log_std_max=2, target_entropy=None,
                 critic_fit_params=None, buffer_strategy="uniform", buffer_alpha=None, buffer_beta=None):
        """
        Constructor.
        Args:
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigm
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.
            buffer_strategy (string): Buffer sampling strategy. One of [uniform, prioritzed, uncertainty]
        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)
        self._reward_scale = 1
        self._buffer_strategy = buffer_strategy
        self._buffer_alpha = 1 if buffer_alpha is None else 1.0
        self._buffer_beta = 0.9 if buffer_beta is None else 0.9

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy
        if self._buffer_strategy == "uniform":
            self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        elif self._buffer_strategy == "prioritized":
            self._replay_memory = PrioritizedReplayMemory(initial_replay_size, max_replay_size, alpha=self._buffer_alpha, beta=self._buffer_beta)
        
        else:
            self._replay_memory = UncertaintyReplayMemory(initial_replay_size, max_replay_size, alpha=self._buffer_alpha, beta=self._buffer_beta)
        """
        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2
        """
        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(MaskedTorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(MaskedTorchApproximator,
                                                     **target_critic_params)

        actor_mu_approximator = Regressor(TorchApproximator,
                                          **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator,
                                             **actor_sigma_params)

        policy = SACPolicy(actor_mu_approximator,
                           actor_sigma_approximator,
                           mdp_info.action_space.low,
                           mdp_info.action_space.high,
                           log_std_min,
                           log_std_max)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        if self._buffer_strategy == "uniform":
            self._replay_memory.add(dataset)
        elif self._buffer_strategy in ["prioritized", "uncertainty"]:
            self._replay_memory.add(dataset, p=np.ones(shape=(len(dataset,))))
        else:
            raise NotImplementedError("Unknown Sampling Strategy Choose one of [uniform, prioritized, uncertainty]")
        if self._replay_memory.initialized:
            if self._buffer_strategy == "uniform":
                state, action, reward, next_state, absorbing, _ = \
                                                            self._replay_memory.get(self._batch_size())
                # no loss correction
                weight = None
            elif self._buffer_strategy == "uncertainty":
                state, action, reward, next_state, absorbing, _ , num_visits, idx =\
                                                            self._replay_memory.get(self._batch_size())
                weight = 1/num_visits
            else:
                state, action, reward, next_state, absorbing, _, idx, is_weight = \
                                                            self._replay_memory.get(self._batch_size())
                # importance sampling loss correction
                weight = is_weight

            td_pred  = self._critic_approximator.predict(state, action,  **self._critic_fit_params)
            mask = np.array([td_pred[0, ...] !=0], dtype=np.float32)
            
            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob, mask)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())

            q_next = self._next_q(next_state, absorbing)

            q = self._reward_scale * np.repeat(np.expand_dims(reward, axis=1),q_next.shape[-1], axis=1) + self.mdp_info.gamma * q_next
            
            if self._buffer_strategy == "uncertainty":
                critic_prediction = td_pred[:, td_pred[0].nonzero()]
                self._replay_memory.update(np.squeeze(critic_prediction), num_visits = num_visits ,idx=idx)
            elif self._buffer_strategy == "prioritized":
                # choose the sampled heads only
                critic_prediction = np.squeeze(td_pred[:, td_pred[0].nonzero()])
                q_heads = np.squeeze(q[:, td_pred[0].nonzero()])
                td_error = np.mean(np.square(critic_prediction - q_heads), axis=1)
                self._replay_memory.update(np.squeeze(td_error), idx=idx)
            
            
            self._critic_approximator.fit(state, action, q, weights=weight,
                                          **self._critic_fit_params)


            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def _loss(self, state, action_new, log_prob, mask):
        q = self._critic_approximator(state, action_new,
                                        output_tensor=True)
        q = torch.squeeze(q[:, mask])
        #print(f"q min {q[0, ...]} mask {mask}")

        q = torch.sum(q, dim=1) / torch.sum(torch.from_numpy(mask))

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.
        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.
        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(next_state, a)
        log_prob_next = np.repeat(np.expand_dims(log_prob_next, axis=1), q.shape[-1], axis=1)
        q = q - self._alpha_np * log_prob_next
        q *= np.ones(q.shape) -  np.repeat(np.expand_dims(absorbing, axis=1), q.shape[-1], axis=1)

        return q

    def save_buffer_snapshot(self,alg_name, epoch):
        priorities = self._replay_memory.priorities
        fig, ax = plt.subplots(figsize=(10,7))
        ax.bar(np.arange(len(priorities)), height=priorities)
        fig.savefig(f"{alg_name}buffer_prios_epoch{epoch}.png")
        plt.close()

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()