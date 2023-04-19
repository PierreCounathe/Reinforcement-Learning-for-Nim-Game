from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

import constants
from StickGame import Game
import utils


class Agent:
    def __init__(
        self, lr: float = 0.1, gamma: float = 0.9, learning: bool = False
    ) -> None:
        """Initializes an agent who's q-table is filled with zeros. The learning boolean determines
        if the policy is greedy (when learning=False) or epsilon-greedy, and if the q-table should be updated.
        """
        self.q_table = np.zeros(
            (constants.MAX_STATE_NUMBER + 1, constants.MAX_ACTION_NUMBER + 1)
        )
        self.training_epochs = 0
        self.lr = lr
        self.gamma = gamma
        self.learning = learning

    def epsilon_greedy_policy(self, game: Game, epsilon: float) -> int:
        """Returns a random action with probability epsilon, and the action that maximizes the next state's value
        with probability 1-epsilon."""
        possible_actions = game.get_possible_actions()
        if np.random.random() <= epsilon:
            action = np.random.choice(possible_actions)
        else:
            current_state_q_values = self.q_table[game.get_state_int_encoded()]
            mask = np.ones(current_state_q_values.size, dtype=bool)
            mask[possible_actions] = False
            current_state_q_values[mask] = -float("inf")
            set_of_best_actions = np.argwhere(
                current_state_q_values == np.amax(current_state_q_values)
            ).flatten()
            action = np.random.choice(set_of_best_actions)
        return action

    def update_q_value(
        self,
        state_int_encoding: int,
        action: int,
        reward: int,
        new_state_int_encoding: Optional[int] = None,
    ) -> None:
        """Updates the q_value of the state-action pair that was just used. When the action taken is forbidden,
        there is no new state to compute the expected discounted next states's reward, it is replaced by 0.
        """
        if new_state_int_encoding:
            self.q_table[state_int_encoding, action] = self.q_table[
                state_int_encoding, action
            ] + self.lr * (
                reward
                - self.gamma * max(self.q_table[new_state_int_encoding])
                - self.q_table[state_int_encoding, action]
            )
        else:  # If no new_state is provided (the action is invalid) update without discounted reward of new state
            self.q_table[state_int_encoding, action] = self.q_table[
                state_int_encoding, action
            ] + self.lr * (reward - self.q_table[state_int_encoding, action])

    def choose_action(self, game: Game) -> Tuple[int, int, int]:
        """Sets epsilon for the epsilon-greedy policy and returns the action, line and sticks Integers."""
        if self.learning:
            epsilon = 0.1 + (self.training_epochs + 1) ** (
                -0.8
            )  # decay epsilon with training_epochs
        else:
            epsilon = 0
        action = self.epsilon_greedy_policy(game, epsilon)
        line, sticks = (action // 10, action % 10)
        return action, line, sticks

    def td_learning(self, game: Game) -> Tuple[int, int, int, int]:
        """Temporal Difference Learning: given the state and q_table, plays an action. If not valid and the
        Agent is learning, update the state-action pair value until the played action is valid. If the Agent
        is learning, update the valid state-action pair value. Returns the line and sticks integers as well as
        the play reward and the play's number of fails before selecting a valid action.
        """
        state_int_encoding = game.get_state_int_encoded()
        action, line, sticks = self.choose_action(game)
        action_reward = 0
        action_fails = 0
        while not game.is_valid_play(line, sticks):
            reward = -10
            action_reward += reward
            action_fails += 1
            if self.learning:
                self.update_q_value(state_int_encoding, action, reward)
            action, line, sticks = self.choose_action(game)
        # Now the taken action is valid
        game.play(line, sticks)
        new_state_int_encoding = game.get_state_int_encoded()
        if sum(game.state) == 1:
            reward = 1
        elif sum(game.state) == 0:
            reward = -1
        else:
            reward = 0
        action_reward += reward
        if self.learning:
            self.update_q_value(
                state_int_encoding, action, reward, new_state_int_encoding
            )
        return line, sticks, action_reward, action_fails

    def increment_training_epochs(self) -> None:
        """Increments the training_epochs field."""
        self.training_epochs += 1

    def copy(self, other_agent: Agent, learning: bool = False) -> None:
        """Copies the q_table, training_epochs and training hyperparameters from another Agent object.
        Precises if the new instance is learning or not."""
        self.q_table = other_agent.q_table
        self.training_epochs = other_agent.training_epochs
        self.lr = other_agent.lr
        self.gamma = other_agent.gamma
        self.learning = learning

    def save(self, name: str) -> None:
        """Saves the current agent with the name and training_epochs infos."""
        utils.check_create_folder("cache/")
        filename = f"cache/agent-{name}-{self.training_epochs}-epochs.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, name: str, training_epochs: str) -> None:
        """Loads a cached agent using its name and training_epochs."""
        filename = f"cache/agent-{name}-{training_epochs}-epochs.pkl"
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
            self.training_epochs = int(training_epochs)
