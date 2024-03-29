from asyncio.log import logger
import sys
from matplotlib.pyplot import axis
import numpy as np
from collections import deque, namedtuple

from mushroom_rl.core import Serializable
from mushroom_rl.utils.parameters import to_parameter

def stable_softmax(x):
    z = x - np.amax(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator

    return softmax

class UncertaintySumTree(object):
    """
    This class implements a sum tree data structure.
    This is used, for instance, by ``PrioritizedReplayMemory``.

    """
    def __init__(self, max_size):
        """
        Constructor.

        Args:
            max_size (int): maximum size of the tree.

        """
        self._max_size = max_size
        self._tree = np.zeros(2 * max_size - 1)
        self._data = [None for _ in range(max_size)]
        self._idx = 0
        self._full = False

    def add(self, dataset, priority, n_steps_return, gamma):
        """
        Add elements to the tree.
        Args:
            dataset (list): list of elements to add to the tree;
            priority (np.ndarray): priority of each sample in the dataset;
            n_steps_return (int): number of steps to consider for computing n-step return;
            gamma (float): discount factor for n-step return.
        """
        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset[i][2]

            j = 0
            while j < n_steps_return - 1:
                if dataset[i + j][5]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset[i + j][2]
            else:
                d = list(dataset[i])
                d[2] = reward
                d[3] = dataset[i + j][3]
                d[4] = dataset[i + j][4]
                d[5] = dataset[i + j][5]
                d.append(0)
                idx = self._idx + self._max_size - 1

                self._data[self._idx] = d
                self.update([idx], [priority[i]])

                self._idx += 1
                if self._idx == self._max_size:
                    self._idx = 0
                    self._full = True

                i += 1

    def get(self, s):
        """
        Returns the provided number of states from the replay memory.

        Args:
            s (float): the value of the samples to return.

        Returns:
            The requested sample.

        """
        idx = self._retrieve(s, 0)
        data_idx = idx - self._max_size + 1
        self._data[data_idx][6] += 1 # Update the num visits of the sampled state

        return idx, self._tree[idx], self._data[data_idx]

    def update(self, idx, priorities):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            idx (np.ndarray): indexes of the transitions in the dataset;
            priorities (np.ndarray): priorities of the transitions.

        """
        for i, p in zip(idx, priorities):
            delta = p - self._tree[i]

            self._tree[i] = p
            self._propagate(delta, i)

    def _propagate(self, delta, idx):
        parent_idx = (idx - 1) // 2

        self._tree[parent_idx] += delta

        if parent_idx != 0:
            self._propagate(delta, parent_idx)

    def _retrieve(self, s, idx):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if self._tree[left] == self._tree[right]:
            return self._retrieve(s, np.random.choice([left, right]))

        if s <= self._tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self._tree[left], right)

    @property
    def size(self):
        """
        Returns:
            The current size of the tree.

        """
        return self._idx if not self._full else self._max_size

    @property
    def max_p(self):
        """
        Returns:
            The maximum priority among the ones in the tree.

        """
        return self._tree[-self._max_size:].max()

    @property
    def total_p(self):
        """
        Returns:
            The sum of the priorities in the tree, i.e. the value of the root
            node.

        """
        return self._tree[0]

class UncertaintyReplayMemory(Serializable):
    """
    This class implements function to manage a prioritized replay memory as the
    one used in "Prioritized Experience Replay" by Schaul et al., 2015.

    """
    def __init__(self, initial_size, max_size, alpha, beta, epsilon=.01):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay
                memory;
            max_size (int): maximum number of elements that the replay memory
                can contain;
            alpha (float): prioritization coefficient;
            beta ([float, Parameter]): importance sampling coefficient;
            epsilon (float, .01): small value to avoid zero probabilities.

        """
        self._priorities = deque([], maxlen=max_size)
        self._initial_size = initial_size
        self._max_size = max_size
        self._max_mean_track = [1,0]
        self._min_mean_track = [0,0]
        self._max_std_track = [1,0]
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon
        self._max_variance = 1
        self._max_mean = 1
        self._max_priority = 1

        self._tree = UncertaintySumTree(max_size)

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _alpha='primitive',
            _beta='primitive',
            _epsilon='primitive',
            _tree='pickle!',
            _priorities='pickle!'
        )

    def add(self, dataset, p, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
            p (np.ndarray): priority of each sample in the dataset.
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        assert n_steps_return > 0
        #maximum priority for new samples
        p *= self.max_priority
        for prio in p:
            self._priorities.append(prio) 
        self._tree.add(dataset, p, n_steps_return, gamma)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        states = [None for _ in range(n_samples)]
        actions = [None for _ in range(n_samples)]
        rewards = [None for _ in range(n_samples)]
        next_states = [None for _ in range(n_samples)]
        absorbing = [None for _ in range(n_samples)]
        last = [None for _ in range(n_samples)]
        num_visits = [None for _ in range(n_samples)]

        idxs = np.zeros(n_samples, dtype=np.int)
        priorities = np.zeros(n_samples)

        total_p = self._tree.total_p
        segment = total_p / n_samples

        a = np.arange(n_samples) * segment
        b = np.arange(1, n_samples + 1) * segment
        if a.any() == np.nan or b.any() == np.nan:
            print(f"a {a}, b {b}")

        samples = np.random.uniform(a, b)
        for i, s in enumerate(samples):
            idx, p, data = self._tree.get(s)
            idxs[i] = idx
            priorities[i] = p
            states[i], actions[i], rewards[i], next_states[i], absorbing[i],\
                last[i], num_visits[i] = data
            states[i] = np.array(states[i])
            next_states[i] = np.array(next_states[i])

        sampling_probabilities = priorities / self._tree.total_p

        return np.array(states), np.array(actions), np.array(rewards),\
            np.array(next_states), np.array(absorbing), np.array(last),\
            np.array(num_visits),idxs

    def update(self, critic_prediction, num_visits , idx):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            critic_prediction (np.ndarray): critic predictions to consider to compute the priorities;
            num visists (np.ndarray): count how often a state has been seen
            idx (np.ndarray): indexes of the transitions in the dataset.

        """
        p = self._get_priority(critic_prediction, num_visits, idx)
        self._update_priorites(idx,p)
        self._tree.update(idx, p)

    def _get_priority(self, critic_prediction, num_visits, idx):
        mean = np.mean(critic_prediction, axis=-1)
        std = np.std(critic_prediction, axis=-1)
        min_mean = np.amin(mean)
        if self._min_mean_track[0] > min_mean or self._min_mean_track[1] in idx:
            self._min_mean_track[0] = min_mean 
            self._min_mean_track[1] = idx[np.argmin(mean)]
        
        mean += np.abs(self._min_mean_track[0])
        max_mean = np.amax(mean)
        max_std = np.amax(std)

        # Normalize mean and variance
        mean /= self.max_mean
        std /= self.max_variance

        #keep track of max values for normalisation
        if self._max_mean_track[0] < max_mean or self._max_mean_track[1] in idx:
            self._max_mean_track[0] = max_mean 
            self._max_mean_track[1] = idx[np.argmax(mean)]
        
        if self._max_std_track[0] < max_std or self._max_std_track[1] in idx:
            self._max_std_track[0] = max_std
            self._max_std_track[1] = idx[np.argmax(std)]
        # sanity check if num visits is not updated
        num_visits[num_visits==0] = 1
        scale = 1 - 1/num_visits  
        priorities = (std/num_visits + scale *std *mean)** self._alpha
        # print(f"min{np.amin(num_visits)} max {np.amax(num_visits)}")
        return priorities 

    def _update_priorites(self, idx, p):
        for i , index in enumerate(idx):
            self._priorities[index-self._max_size] = p[i]
    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self._tree.size > self._initial_size

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.

        """
        return self._tree.max_p if self.initialized else 1.

    @property
    def max_variance(self):
        return self._max_std_track[0]

    @property
    def max_mean(self):
        return self._max_mean_track[0]
    
    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.
        """
        return self._tree.size

    @property
    def priorities(self):
        return self._priorities

    def _post_load(self):
        if self._tree is None:
            self._tree = UncertaintySumTree(self._max_size)


class AlternativeMEETReplayMemory(Serializable):
    """
    This class implements function to manage a prioritized replay memory as the
    one used in "Prioritized Experience Replay" by Schaul et al., 2015.

    """
    def __init__(self, initial_size, max_size, alpha, beta, epsilon=.01):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the replay
                memory;
            max_size (int): maximum number of elements that the replay memory
                can contain;
            alpha (float): prioritization coefficient;
            beta ([float, Parameter]): importance sampling coefficient;
            epsilon (float, .01): small value to avoid zero probabilities.

        """
        self._priorities = deque([], maxlen=max_size)
        self._buffer = deque([], maxlen=max_size)
        self._mean_track = [0,0]
        self._std_track = [0,0]
        self._min_mean_track = [0,0]
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon
        self._max_variance = 1
        self._max_mean = 1
        self._max_priority = 1

    def add(self, dataset, priority, n_steps_return=1, gamma=0.99):
        """
        Add elements to the tree.

        Args:
            dataset (list): list of elements to add to the tree;
            priority (np.ndarray): priority of each sample in the dataset;
            n_steps_return (int): number of steps to consider for computing n-step return;
            gamma (float): discount factor for n-step return.

        """
        i = 0
        while i < len(dataset) - n_steps_return + 1:
            reward = dataset[i][2]

            j = 0
            while j < n_steps_return - 1:
                if dataset[i + j][5]:
                    i += j + 1
                    break
                j += 1
                reward += gamma ** j * dataset[i + j][2]
            else:
                d = list(dataset[i])
                d[2] = reward
                d[3] = dataset[i + j][3]
                d[4] = dataset[i + j][4]
                d[5] = dataset[i + j][5]
                d.append(0) # add num visits here
                # idx = self._idx + self._max_size - 1

                self._buffer.append(d)
                self._priorities.append(self.max_priority)
                """
                self.update([idx], [priority[i]])

                self._idx += 1
                if self._idx == self._max_size:
                    self._idx = 0
                    self._full = True
                """
                i += 1

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        states = [None for _ in range(n_samples)]
        actions = [None for _ in range(n_samples)]
        rewards = [None for _ in range(n_samples)]
        next_states = [None for _ in range(n_samples)]
        absorbing = [None for _ in range(n_samples)]
        last = [None for _ in range(n_samples)]
        num_visits = [None for _ in range(n_samples)]

        idxs = [None for _ in range(n_samples)]
        priorities = stable_softmax(np.array(self.priorities))

        

        sample_indices = np.random.choice(np.arange(0,self.size), size=n_samples, p = priorities)
        for i, idx in enumerate(sample_indices):
            states[i], actions[i], rewards[i], next_states[i], absorbing[i],\
                last[i], num_visits[i] = self._buffer[idx]
            num_visits[i] += 1
            states[i] = np.array(states[i])
            next_states[i] = np.array(next_states[i])
            idxs[i] = idx

        return np.array(states), np.array(actions), np.array(rewards),\
            np.array(next_states), np.array(absorbing), np.array(last),\
            np.array(num_visits), idxs

    def update(self, critic_prediction, num_visits , idx):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            critic_prediction (np.ndarray): critic predictions to consider to compute the priorities;
            num visists (np.ndarray): count how often a state has been seen
            idx (np.ndarray): indexes of the transitions in the dataset.

        """
        p = self._get_priority(critic_prediction, num_visits, idx)
        self._update_priorities(idx, p)

    def _update_priorities(self, idx, p):
        for i, index in enumerate(idx):
            self._priorities[index] = p[i]
            

    def _get_priority(self, critic_prediction, num_visits, idx):
        mean = np.mean(critic_prediction, axis=-1) 
        std = np.std(critic_prediction, axis=-1)

        max_mean = np.amax(mean)
        min_mean = np.amin(mean)
        max_std = np.amax(std)
        #keep track of max values for normalisation
        if self._mean_track[0] < max_mean or self._mean_track[1] in idx:
            self._mean_track[0] = max_mean 
            self._mean_track[1] = idx[np.argmax(mean)]

        if self._min_mean_track[0] > min_mean or self._min_mean_track[1] in idx:
            self._min_mean_track[0] = min_mean 
            self._min_mean_track[1] = idx[np.argmin(mean)]
        
        if self._std_track[0] < max_std or self._std_track[1] in idx:
            self._std_track[0] = max_std
            self._std_track[1] = idx[np.argmax(std)]

        # Normalize mean and variance
        mean += self._min_mean_track[0]
        mean /= self.max_mean
        std /= self.max_variance
        # sanity check if num visits is not updated
        num_visits[num_visits==0] = 1
        beta = 1 - 1/num_visits  
        priorities = std/num_visits + beta *std *mean
        print(f"min{np.amin(priorities)} max {np.amax(priorities)}")
        return priorities 

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return len(self._buffer) > self._initial_size

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.

        """
        return self._max_priority

    @property
    def max_variance(self):
        return self._std_track[0]

    @property
    def max_mean(self):
        return self._mean_track[0]

    
    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.
        """
        return len(self._buffer)

    @property
    def priorities(self):
        return self._priorities
    