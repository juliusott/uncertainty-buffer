import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
import torch
from approximators.masked_torch_regressor import  MaskedTorchApproximator
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory, PrioritizedReplayMemory
from buffer.uncertainty_buffer import UncertaintyReplayMemory
from mushroom_rl.utils.parameters import Parameter, to_parameter
from copy import deepcopy
import torch.nn.functional as F

class MultiHeadDDPG(DeepAC):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.
    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, policy_delay=1,
                 critic_fit_params=None, actor_predict_params=None, critic_predict_params=None):
        """
        Constructor.
        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ((float, Parameter)): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 1): the number of updates of the critic after
                which an actor update is implemented;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;
            actor_predict_params (dict, None): parameters for the prediction with the
                actor approximator;
            critic_predict_params (dict, None): parameters for the prediction with the
                critic approximator.
        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._policy_delay = to_parameter(policy_delay)
        self._fit_count = 0

        self._replay_memory = UncertaintyReplayMemory(initial_replay_size, max_replay_size, alpha=0.1, beta=0.9)

        self.n_heads = critic_params["output_shape"][0]
        self.critic = critic_params["network"]

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(MaskedTorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(MaskedTorchApproximator,
                                                     **target_critic_params)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator,
                                                    **target_actor_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)
        self._init_target(self._actor_approximator,
                          self._target_actor_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _policy_delay='mushroom',
            _fit_count='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _target_actor_approximator='mushroom'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        self._replay_memory.add(dataset, p=np.ones(shape=(len(dataset,))))
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ , num_visits, idx, _=\
                self._replay_memory.get(self._batch_size())

            q_next = self._next_q(next_state, absorbing)

            q = np.repeat(np.expand_dims(reward, axis=1),self.n_heads, axis=1) + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            td_pred  = self._critic_approximator.predict(state, action,  **self._critic_fit_params)
            #print(f"td pred {td_pred[0]} q {q[0]}")
            critic_prediction = td_pred[:, td_pred[0].nonzero()]

            td_error = td_pred[:, td_pred[0].nonzero()] - q[:, td_pred[0].nonzero()]
            #print(f"td pred {td_pred[:, td_pred[0].nonzero()][0]} q {q[:, td_pred[0].nonzero()][0]}")
            td_error = np.squeeze(td_error)
            self._replay_memory.update(np.squeeze(critic_prediction), num_visits = num_visits ,idx=idx)
            if self._fit_count % self._policy_delay() == 0:
                loss = self._loss(state)
                self._optimize_actor_parameters(loss)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)
            self._update_target(self._actor_approximator,
                                self._target_actor_approximator)

            self._fit_count += 1

    def _loss(self, state):
        action = self._actor_approximator(state, output_tensor=True, **self._actor_predict_params)
        q = self._critic_approximator(state, action, output_tensor=True, **self._critic_predict_params)

        return -q.mean()

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
        a = self._target_actor_approximator.predict(next_state, **self._actor_predict_params)

        q = self._target_critic_approximator.predict(next_state, a, **self._critic_predict_params)
        q *= np.ones(q.shape) -  np.repeat(np.expand_dims(absorbing, axis=1), self.n_heads, axis=1)

        return q

    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())