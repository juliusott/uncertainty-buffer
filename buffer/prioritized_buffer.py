import operator
import random
from enum import Enum
from typing import List, Tuple, Any

import numpy as np

from mushroom_rl.core import Serializable
from mushroom_rl.utils.parameters import to_parameter



class SumTree(object):
    """
    A tree which can be used as a min/max heap or a sum tree
    Add or update item value - O(log N)
    Sampling an item - O(log N)
    """
    class Operation(Enum):
        MAX = {"operator": max, "initial_value": -float("inf")}
        MIN = {"operator": min, "initial_value": float("inf")}
        SUM = {"operator": operator.add, "initial_value": 0}

    def __init__(self, size: int, operation: Operation):
        self.next_leaf_idx_to_write = 0
        self.size = size
        if not (size > 0 and size & (size - 1) == 0):
            raise ValueError("A segment tree size must be a positive power of 2. The given size is {}".format(self.size))
        self.operation = operation
        self.tree = np.ones(2 * size - 1) * self.operation.value['initial_value']
        self.data = [None] * size

    def _propagate(self, node_idx: int) -> None:
        """
        Propagate an update of a node's value to its parent node
        :param node_idx: the index of the node that was updated
        :return: None
        """
        parent = (node_idx - 1) // 2

        self.tree[parent] = self.operation.value['operator'](self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])

        if parent != 0:
            self._propagate(parent)

    def _retrieve(self, root_node_idx: int, val: float)-> int:
        """
        Retrieve the first node that has a value larger than val and is a child of the node at index idx
        :param root_node_idx: the index of the root node to search from
        :param val: the value to query for
        :return: the index of the resulting node
        """
        left = 2 * root_node_idx + 1
        right = left + 1

        if left >= len(self.tree):
            return root_node_idx

        if val <= self.tree[left]:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val-self.tree[left])

    @property
    def total_p(self) -> float:
        """
        Return the total value of the tree according to the tree operation. For SUM for example, this will return
        the total sum of the tree. for MIN, this will return the minimal value
        :return: the total value of the tree
        """
        return self.tree[0]

    def add(self, val: float, data: Any) -> None:
        """
        Add a new value to the tree with data assigned to it
        :param val: the new value to add to the tree
        :param data: the data that should be assigned to this value
        :return: None
        """
        self.data[self.next_leaf_idx_to_write] = data
        self.update(self.next_leaf_idx_to_write, val)

        self.next_leaf_idx_to_write += 1
        if self.next_leaf_idx_to_write >= self.size:
            self.next_leaf_idx_to_write = 0

    def update(self, leaf_idx: int, new_val: float) -> None:
        """
        Update the value of the node at index idx
        :param leaf_idx: the index of the node to update
        :param new_val: the new value of the node
        :return: None
        """
        node_idx = leaf_idx + self.size - 1
        if not 0 <= node_idx < len(self.tree):
            raise ValueError("The given left index ({}) can not be found in the tree. The available leaves are: 0-{}"
                             .format(leaf_idx, self.size - 1))

        self.tree[node_idx] = new_val
        self._propagate(node_idx)

    def get_element_by_partial_sum(self, val: float) -> Tuple[int, float, Any]:
        """
        Given a value between 0 and the tree sum, return the object which this value is in it's range.
        For example, if we have 3 leaves: 10, 20, 30, and val=35, this will return the 3rd leaf, by accumulating
        leaves by their order until getting to 35. This allows sampling leaves according to their proportional
        probability.
        :param val: a value within the range 0 and the tree sum
        :return: the index of the resulting leaf in the tree, its probability and
                 the object itself
        """
        node_idx = self._retrieve(0, val)
        leaf_idx = node_idx - self.size + 1
        data_value = self.tree[node_idx]
        data = self.data[leaf_idx]

        return leaf_idx, data_value, data

    def __str__(self):
        result = ""
        start = 0
        size = 1
        while size <= self.size:
            result += "{}\n".format(self.tree[start:(start + size)])
            start += size
            size *= 2
        return result


class PrioritizedReplayMemory(Serializable):
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
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon
        self._max_priority = 1
        self._num_transitions = 0
        
        #self.power_of_2_size = 1
        #while self.power_of_2_size < max_size:
        #    self.power_of_2_size *= 2
        self.power_of_2_size = max_size

        self._tree = SumTree(self.power_of_2_size, operation=SumTree.Operation.SUM)
        self._min_tree = SumTree(self.power_of_2_size, SumTree.Operation.MIN)
        self._max_tree = SumTree(self.power_of_2_size, SumTree.Operation.MAX)

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _alpha='primitive',
            _beta='primitive',
            _epsilon='primitive',
            _tree='pickle!'
        )

    def add(self, dataset, p):
        """
        Add elements to the replay memory.
        Args:
            dataset (list): list of elements to add to the replay memory;
        """
        for data in dataset:
            self._tree.add(self._max_priority, data)
        self._num_transitions += len(dataset)

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

        idxs = np.zeros(n_samples, dtype=np.int)
        priorities = np.zeros(n_samples)

        total_p = self._tree.total_p
        segment_size = total_p / n_samples
        min_probability = self.min_priority / self._tree.total_p # min P(j) = min p^a / sum(p^a)
        max_weight = (min_probability * self.size) ** -self._beta()  # max wi

        for i in range(n_samples):
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)
            val = np.random.uniform(segment_start, segment_end)
            idx, priority, data = self._tree.get_element_by_partial_sum(val)

            idxs[i] = idx
            priorities[i] = priority
            states[i], actions[i], rewards[i], next_states[i], absorbing[i],\
                last[i] = data
            states[i] = np.array(states[i])
            next_states[i] = np.array(next_states[i])

        sampling_probabilities = priorities / self._tree.total_p
        is_weight = (self._tree.size * sampling_probabilities) ** -self._beta()
        is_weight /= max_weight

        return np.array(states), np.array(actions), np.array(rewards),\
            np.array(next_states), np.array(absorbing), np.array(last),\
            idxs, is_weight

    def update(self, error, idx):
        """
        Update the priority of the sample at the provided index in the dataset.
        Args:
            error (np.ndarray): errors to consider to compute the priorities;
            idx (np.ndarray): indexes of the transitions in the dataset.
        """
        if error.any() < 0:
            raise ValueError("The priorities must be non-negative values")
        priority = (error + self._epsilon)
        #print(priority**self._alpha)
        for prio, index in zip(priority, idx):
            self._tree.update(index, prio ** self._alpha)
            self._min_tree.update(index, prio ** self._alpha)
            self._max_tree.update(index, prio)
            self._max_priority = self._max_tree.total_p


    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.
        """
        return self._tree.size > self._initial_size

    @property
    def size(self):
        return min(self._num_transitions, self._max_size)

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.
        """
        return self._max_priority

    @property
    def min_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.
        """
        return self._min_tree.total_p

    def _post_load(self):
        if self._tree is None:
            self._tree = SumTree(self._max_size)


