
"""
ronan kenny

"""
from src.CoalitionPolicyRegistry import CoalitionPolicyRegistry
import itertools
import math
import numpy as np

class ACDEvaluator:
    """
    Computes the Average Counterfactual Difference Characteristic Value Function:
      - V-mode (Eq. 9): v_ACD^V(C) = (1/|A|) * sum_a [ V^{pi^C}(s*^C) - V^{pi^C}(s'^C) ]
      - Q-mode (Eq. 8): v_ACD^Q(C) = (1/|A|) * sum_a [ Q^{pi^C}(s^C, a*) - Q^{pi^C}(s^C, a) ]
    """

    def __init__(self, env, registry: CoalitionPolicyRegistry, mode: str = "V"):
        assert mode in ("V", "Q"), "mode must be 'V' or 'Q'"
        self.env = env
        self.registry = registry
        self.mode = mode

    def _get_next_state_value(self, state, action, subset: tuple) -> float:
        next_state, reward, done = self.env.step(state, action)
        if done:
            # Terminal transition value equals the immediate goal/hole reward
            return float(reward)
        ns_c = self.registry.masker.project(next_state, subset)
        return self.registry.registry[subset]["V"].get(ns_c, 0.0)

    def evaluate(self, state, subset: tuple) -> float:
        if len(subset) == 0:
            return 0.0  # Convention: v(empty) = 0

        full_subset = self.registry.masker.all_features
        # Optimal action a* is always determined by the full-state policy pi^F
        full_s = self.registry.masker.project(state, full_subset)
        a_star = self.registry.registry[full_subset]["policy"][full_s]

        actions = self.env.actions

        if self.mode == "V":
            v_opt = self._get_next_state_value(state, a_star, subset)
            diffs = [v_opt - self._get_next_state_value(state, a, subset) for a in actions]
            return float(np.mean(diffs))

        elif self.mode == "Q":
            s_c = self.registry.masker.project(state, subset)
            q_table = self.registry.registry[subset]["Q"]
            q_opt = float(q_table[s_c][a_star])
            diffs = [q_opt - float(q_table[s_c][a]) for a in actions]
            return float(np.mean(diffs))