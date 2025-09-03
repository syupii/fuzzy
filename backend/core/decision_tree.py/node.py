from typing import Dict, List, Optional, Any
from ..fuzzy.membership import MembershipFunction

class FuzzyDecisionNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.feature_name: Optional[str] = None
        self.membership_functions: Dict[str, MembershipFunction] = {}
        self.children: Dict[str, 'FuzzyDecisionNode'] = {}
        self.is_leaf = False
        self.output_value: Optional[float] = None
        self.confidence: float = 0.0
    
    def predict(self, features: Dict[str, float]) -> float:
        if self.is_leaf:
            return self.output_value
        
        # ファジィ推論による予測
        activations = {}
        for label, mf in self.membership_functions.items():
            if self.feature_name in features:
                activations[label] = mf.membership(features[self.feature_name])
        
        # 子ノードからの予測を重み付け統合
        predictions = []
        weights = []
        
        for label, child in self.children.items():
            if label in activations:
                weight = activations[label]
                if weight > 0:
                    pred = child.predict(features)
                    predictions.append(pred)
                    weights.append(weight)
        
        if not predictions:
            return 0.5  # デフォルト値
        
        # 重み付き平均
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5