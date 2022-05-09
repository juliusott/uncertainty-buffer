from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory, ReplayMemory

from approximators.masked_torch_regressor import MaskedTorchApproximator
from buffer.uncertainty_buffer import UncertaintyReplayMemory


class MultiHeadDDPG(DeepAC):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.
    """

    def __init__(
        self,
        mdp_info,
        policy_class,
        policy_params,
        actor_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        tau,
        policy_delay=1,
        critic_fit_params=None,
        actor_predict_params=None,
        critic_predict_params=None,
        warmup_transitions=100,
        buffer_strategy="uniform",
    ):
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
        self._critic_fit_params = (
            dict() if critic_fit_params is None else critic_fit_params
        )
        self._actor_predict_params = (
            dict() if actor_predict_params is None else actor_predict_params
        )
        self._critic_predict_params = (
            dict() if critic_predict_params is None else critic_predict_params
        )

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._policy_delay = to_parameter(policy_delay)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._fit_count = 0
        self._reward_scale = 1
        self._buffer_strategy = buffer_strategy

        if self._buffer_strategy == "uniform":
            self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        elif self._buffer_strategy == "prioritized":
            self._replay_memory = PrioritizedReplayMemory(
                initial_replay_size, max_replay_size, alpha=1, beta=0.9
            )

        else:
            self._replay_memory = UncertaintyReplayMemory(
                initial_replay_size, max_replay_size, alpha=1, beta=0.9
            )

        self.n_heads = critic_params["output_shape"][0]
        self.critic = critic_params["network"]

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(MaskedTorchApproximator, **critic_params)
        self._target_critic_approximator = Regressor(
            MaskedTorchApproximator, **target_critic_params
        )

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator, **actor_params)
        self._target_actor_approximator = Regressor(
            TorchApproximator, **target_actor_params
        )

        self._init_target(self._critic_approximator, self._target_critic_approximator)
        self._init_target(self._actor_approximator, self._target_actor_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        self._add_save_attr(
            _critic_fit_params="pickle",
            _critic_predict_params="pickle",
            _actor_predict_params="pickle",
            _batch_size="mushroom",
            _tau="mushroom",
            _policy_delay="mushroom",
            _fit_count="primitive",
            _replay_memory="mushroom",
            _critic_approximator="mushroom",
            _target_critic_approximator="mushroom",
            _target_actor_approximator="mushroom",
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        if self._buffer_strategy == "uniform":
            self._replay_memory.add(dataset)
        elif self._buffer_strategy in ["prioritized", "uncertainty"]:
            self._replay_memory.add(
                dataset,
                p=np.ones(
                    shape=(
                        len(
                            dataset,
                        )
                    )
                ),
            )
        else:
            raise NotImplementedError(
                "Unknown Sampling Strategy Choose one of [uniform, prioritized, uncertainty]"
            )
        if self._replay_memory.initialized:
            if self._buffer_strategy == "uniform":
                (
                    state,
                    action,
                    reward,
                    next_state,
                    absorbing,
                    _,
                ) = self._replay_memory.get(self._batch_size())
                # no loss correction
                weight = None
            elif self._buffer_strategy == "uncertainty":
                (
                    state,
                    action,
                    reward,
                    next_state,
                    absorbing,
                    _,
                    num_visits,
                    idx,
                ) = self._replay_memory.get(self._batch_size())
                weight = 1 / num_visits
            else:
                (
                    state,
                    action,
                    reward,
                    next_state,
                    absorbing,
                    _,
                    idx,
                    is_weight,
                ) = self._replay_memory.get(self._batch_size())
                # importance sampling loss correction
                weight = is_weight

            q_next = self._next_q(next_state, absorbing)

            q = (
                self._reward_scale
                * np.repeat(np.expand_dims(reward, axis=1), self.n_heads, axis=1)
                + self.mdp_info.gamma * q_next
            )

            self._critic_approximator.fit(
                state, action, q, weights=weight, **self._critic_fit_params
            )

            td_pred = self._critic_approximator.predict(
                state, action, **self._critic_fit_params
            )
            mask = torch.from_numpy(np.array([td_pred[0, ...] != 0], dtype=np.float32))
            if self._buffer_strategy == "uncertainty":
                td_pred = self._critic_approximator.predict(
                    state, action, **self._critic_fit_params
                )
                critic_prediction = td_pred[:, td_pred[0].nonzero()]
                self._replay_memory.update(
                    np.squeeze(critic_prediction), num_visits=num_visits, idx=idx
                )
            elif self._buffer_strategy == "prioritized":
                td_pred = self._critic_approximator.predict(
                    state, action, **self._critic_fit_params
                )
                # choose the sampled heads only
                critic_prediction = np.squeeze(td_pred[:, td_pred[0].nonzero()])
                q_heads = np.squeeze(q[:, td_pred[0].nonzero()])
                td_error = np.mean(np.square(critic_prediction - q_heads), axis=1)

                self._replay_memory.update(np.squeeze(td_error), idx=idx)

            if self._fit_count % self._policy_delay() == 0:
                loss = self._loss(state, mask)
                self._optimize_actor_parameters(loss)

            self._update_target(
                self._critic_approximator, self._target_critic_approximator
            )
            self._update_target(
                self._actor_approximator, self._target_actor_approximator
            )

            self._fit_count += 1

    def _loss(self, state, mask):
        action = self._actor_approximator(
            state, output_tensor=True, **self._actor_predict_params
        )
        q = self._critic_approximator(
            state, action, output_tensor=True, **self._critic_predict_params
        )
        q = torch.min(q[:, torch.squeeze(mask) != 0.0], dim=1).values
        # print(f"q {q} mask {mask}")
        # q = torch.min(dim=1).values
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
        a = self._target_actor_approximator.predict(
            next_state, **self._actor_predict_params
        )

        q = self._target_critic_approximator.predict(
            next_state, a, **self._critic_predict_params
        )
        q *= np.ones(q.shape) - np.repeat(
            np.expand_dims(absorbing, axis=1), self.n_heads, axis=1
        )

        return q

    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(
            self._actor_approximator.model.network.parameters()
        )

    def save_buffer_snapshot(self, alg_name, epoch):
        priorities = self._replay_memory.priorities
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(np.arange(len(priorities)), height=priorities)
        fig.savefig(f"{alg_name}buffer_prios_epoch{epoch}.png")
        plt.close()
