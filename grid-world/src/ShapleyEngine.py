"""
ronan kenny
"""
import itertools
import math
import numpy as np
from src.FeatureMasker import FeatureMasker
from src.ACDEvaluator import ACDEvaluator

class ShapleyEngine:
    """
    Model-agnostic exact Shapley attribution engine.
    Computes phi_i for each feature i in F = {0, ..., n-1}.
    """

    def __init__(self, masker: FeatureMasker):
        self.masker = masker
        self.n = masker.num_features
        self.features = masker.all_features

    def explain(self, state, acd_evaluator: ACDEvaluator) -> np.ndarray:
        # 1. Pre-evaluate v_ACD(C) for all 2^n subsets once
        cvf_cache = {
            subset: acd_evaluator.evaluate(state, subset)
            for subset in self.masker.all_subsets(include_empty=True)
        }

        # 2. Apply Equation (5) for each feature i
        phi = np.zeros(self.n, dtype=np.float64)
        for i in self.features:
            others = [f for f in self.features if f != i]
            for r in range(len(others) + 1):
                for C in itertools.combinations(others, r):
                    C_sorted = tuple(sorted(C))
                    C_with_i = tuple(sorted(C + (i,)))

                    weight = (math.factorial(len(C)) * math.factorial(self.n - len(C) - 1)) / math.factorial(self.n)
                    marginal_contribution = cvf_cache[C_with_i] - cvf_cache[C_sorted]
                    phi[i] += weight * marginal_contribution

        return phi