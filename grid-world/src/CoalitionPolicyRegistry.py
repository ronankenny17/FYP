"""
ronan kenny 
"""

import itertools
import math
import numpy as np
from src.FeatureMasker import FeatureMasker


class CoalitionPolicyRegistry:
    """
    
    Manages the induced policies pi^C and their value tables (V^{pi^C}, Q^{pi^C})
    for every feature subset C. Allows plugging in an already-trained full model.
    for every feature subset C, the module trains an induced policy pi^C
    this is then use in the ACD calcs Vacd(C u i) - Vacd(C)

    """

    def __init__(self, env, masker: FeatureMasker):
        self.env = env
        self.masker = masker
        # Stores {subset_C: {"policy": dict, "V": dict, "Q": dict}}
        self.registry = {}

    def register_pretrained_full_model(self, policy_dict: dict, V_dict: dict, Q_dict: dict):
        """Plug in your existing trained full-state model for C = F."""
        full_subset = self.masker.all_features
        self.registry[full_subset] = {
            "policy": policy_dict,
            "V": V_dict,
            "Q": Q_dict,
        }

    def train_induced_policy(
        self, subset: tuple, episodes: int = 5000, alpha: float = 0.1, gamma: float = 0.95, eps: float = 0.2
    ):
        """
        Trains an induced policy pi^C via tabular Q-learning Model where the agent
        only observes features in `subset`.
        """
        if len(subset) == 0:
            # Paper convention (Section 3.2): empty set has 0 value
            self.registry[()] = {"policy": {}, "V": {(): 0.0}, "Q": {}}
            return

        Q = {}
        actions = self.env.actions
        valid_starts = self.env.get_non_terminal_states()

        for _ in range(episodes):
            state = valid_starts[np.random.randint(len(valid_starts))]
            for _ in range(50):  # max steps per episode
                s_c = self.masker.project(state, subset)
                if s_c not in Q:
                    Q[s_c] = np.zeros(len(actions))

                # Epsilon-greedy exploration
                if np.random.rand() < eps:
                    action = int(np.random.choice(actions))
                else:
                    action = int(np.argmax(Q[s_c]))

                next_state, reward, done = self.env.step(state, action)
                ns_c = self.masker.project(next_state, subset)
                if ns_c not in Q:
                    Q[ns_c] = np.zeros(len(actions))

                target = reward if done else reward + gamma * float(np.max(Q[ns_c]))
                Q[s_c][action] += alpha * (target - Q[s_c][action])

                state = next_state
                if done:
                    break

        policy = {s_c: int(np.argmax(q_vals)) for s_c, q_vals in Q.items()}
        V = {s_c: float(np.max(q_vals)) for s_c, q_vals in Q.items()}
        self.registry[subset] = {"policy": policy, "V": V, "Q": Q}

    def build_all(self, episodes: int = 5000, alpha: float = 0.1, gamma: float = 0.95, eps: float = 0.2):
        """Trains any coalition subsets not yet registered."""
        for subset in self.masker.all_subsets(include_empty=True):
            if subset not in self.registry:
                self.train_induced_policy(subset, episodes, alpha, gamma, eps)