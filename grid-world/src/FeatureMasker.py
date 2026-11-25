"""
ronan kenny


"""

import itertools
import math
import numpy as np
from collections.abc import Callable

class FeatureMasker:
    """
    Projects a full state s = (x_0, x_1, ..., x_{n-1}) onto a subset of
    feature indices C (e.g., C = (0,) keeps only row x; C = (0, 1) keeps both).
    """

    def __init__(
        self,
        num_features: int,
        feature_extractor: Callable[[tuple], tuple] | None = None,
    ):
        self.num_features = num_features
        self.all_features = tuple(range(num_features))
        self.feature_extractor = feature_extractor

    def project(self, state, subset: tuple) -> tuple:
        if len(subset) == 0:
            return ()
        features = self.feature_extractor(state) if self.feature_extractor else state
        return tuple(features[i] for i in sorted(subset))

    def all_subsets(self, include_empty: bool = True) -> list[tuple]:
        start = 0 if include_empty else 1
        return [
            sub
            for r in range(start, self.num_features + 1)
            for sub in itertools.combinations(self.all_features, r)
        ]