import random
from typing import Dict, Any, List

class TreeConfig:
    def __init__(self):
        self.max_depth = 5
        self.min_samples_leaf = 5

class FuzzyDecisionTree:
    def __init__(self, config: TreeConfig = None):
        self.config = config or TreeConfig()
        self.trained = False
    
    def fit(self, data, target):
        """訓練"""
        self.trained = True
        return self
    
    def predict(self, data):
        """予測"""
        if isinstance(data, list):
            return [random.uniform(0.3, 0.9) for _ in data]
        return random.uniform(0.3, 0.9)

class EnhancedFuzzyDecisionTree(FuzzyDecisionTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

