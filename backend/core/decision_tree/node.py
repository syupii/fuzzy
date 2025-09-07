# core/decision_tree/node.py - ファジィ決定木ノード

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class SplitCondition:
    """分岐条件"""
    feature: str                    # 分岐に使用する特徴量
    threshold: float                # 閾値
    linguistic_value: str           # 言語値（"low", "medium", "high"など）
    membership_threshold: float     # メンバーシップ関数の閾値
    operator: str = ">"            # 比較演算子

class FuzzyTreeNode(ABC):
    """ファジィ決定木ノードの抽象基底クラス"""
    
    def __init__(self, node_id: str, depth: int = 0):
        self.node_id = node_id
        self.depth = depth
        self.parent: Optional['FuzzyTreeNode'] = None
        self.samples_count = 0
        self.purity = 0.0
        
    @abstractmethod
    def predict(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """予測を実行"""
        pass
    
    @abstractmethod
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        pass

class FuzzyInternalNode(FuzzyTreeNode):
    """ファジィ決定木の内部ノード"""
    
    def __init__(self, node_id: str, split_condition: SplitCondition, depth: int = 0):
        super().__init__(node_id, depth)
        self.split_condition = split_condition
        self.children: Dict[str, FuzzyTreeNode] = {}  # 子ノード（"left", "right"など）
        self.split_values: Dict[str, float] = {}      # 各分岐の値
        
    def add_child(self, branch_name: str, child_node: 'FuzzyTreeNode') -> None:
        """子ノードを追加"""
        self.children[branch_name] = child_node
        child_node.parent = self
    
    def predict(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """予測を実行（ファジィ推論）"""
        
        feature_value = sample.get(self.split_condition.feature, 0.0)
        
        # 各子ノードへの帰属度を計算
        membership_values = self._calculate_memberships(feature_value)
        
        # 各子ノードの予測を重み付き平均
        final_prediction = {}
        total_weight = 0.0
        
        for branch_name, membership in membership_values.items():
            if membership > 0 and branch_name in self.children:
                child_prediction = self.children[branch_name].predict(sample)
                
                # 重み付きで統合
                for key, value in child_prediction.items():
                    if key not in final_prediction:
                        final_prediction[key] = 0.0
                    final_prediction[key] += value * membership
                
                total_weight += membership
        
        # 正規化
        if total_weight > 0:
            for key in final_prediction:
                final_prediction[key] /= total_weight
        
        return final_prediction
    
    def _calculate_memberships(self, feature_value: float) -> Dict[str, float]:
        """特徴値に対する各分岐への帰属度を計算"""
        
        memberships = {}
        
        # ファジィ分岐のメンバーシップ関数を適用
        if self.split_condition.linguistic_value == "low":
            # 低値への帰属度
            if feature_value <= self.split_condition.threshold:
                memberships["left"] = 1.0
                memberships["right"] = 0.0
            else:
                # ファジィ境界
                distance = feature_value - self.split_condition.threshold
                max_distance = 3.0  # ファジィ幅
                membership_left = max(0, 1.0 - distance / max_distance)
                memberships["left"] = membership_left
                memberships["right"] = 1.0 - membership_left
                
        elif self.split_condition.linguistic_value == "high":
            # 高値への帰属度
            if feature_value >= self.split_condition.threshold:
                memberships["right"] = 1.0
                memberships["left"] = 0.0
            else:
                # ファジィ境界
                distance = self.split_condition.threshold - feature_value
                max_distance = 3.0
                membership_right = max(0, 1.0 - distance / max_distance)
                memberships["right"] = membership_right
                memberships["left"] = 1.0 - membership_right
                
        else:  # medium
            # 中間値への帰属度（三角形）
            center = self.split_condition.threshold
            width = 2.0
            
            if abs(feature_value - center) <= width:
                membership_center = 1.0 - abs(feature_value - center) / width
                memberships["center"] = membership_center
                memberships["left"] = (1.0 - membership_center) * (1 if feature_value < center else 0)
                memberships["right"] = (1.0 - membership_center) * (1 if feature_value > center else 0)
            else:
                memberships["center"] = 0.0
                if feature_value < center - width:
                    memberships["left"] = 1.0
                    memberships["right"] = 0.0
                else:
                    memberships["left"] = 0.0
                    memberships["right"] = 1.0
        
        return memberships
    
    def is_leaf(self) -> bool:
        """内部ノードは葉ではない"""
        return False
    
    def get_split_info(self) -> Dict[str, Any]:
        """分岐情報を取得"""
        return {
            "feature": self.split_condition.feature,
            "threshold": self.split_condition.threshold,
            "linguistic_value": self.split_condition.linguistic_value,
            "children_count": len(self.children),
            "children": list(self.children.keys())
        }

class FuzzyLeafNode(FuzzyTreeNode):
    """ファジィ決定木の葉ノード"""
    
    def __init__(self, node_id: str, class_distribution: Dict[str, float], 
                 depth: int = 0, confidence: float = 1.0):
        super().__init__(node_id, depth)
        self.class_distribution = class_distribution    # クラス分布
        self.predicted_class = max(class_distribution.keys(), 
                                 key=lambda k: class_distribution[k])  # 予測クラス
        self.confidence = confidence                    # 予測信頼度
        self.support_samples: List[Dict] = []          # サポートサンプル
        
    def predict(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """予測を実行"""
        # 葉ノードのクラス分布をそのまま返す
        return self.class_distribution.copy()
    
    def is_leaf(self) -> bool:
        """葉ノードである"""
        return True
    
    def add_support_sample(self, sample: Dict[str, Any]) -> None:
        """サポートサンプルを追加"""
        self.support_samples.append(sample)
        self.samples_count = len(self.support_samples)
    
    def calculate_entropy(self) -> float:
        """エントロピーを計算"""
        if not self.class_distribution:
            return 0.0
        
        total = sum(self.class_distribution.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in self.class_distribution.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def calculate_gini(self) -> float:
        """ジニ不純度を計算"""
        if not self.class_distribution:
            return 0.0
        
        total = sum(self.class_distribution.values())
        if total == 0:
            return 0.0
        
        gini = 1.0
        for count in self.class_distribution.values():
            probability = count / total
            gini -= probability ** 2
        
        return gini
    
    def get_prediction_info(self) -> Dict[str, Any]:
        """予測情報を取得"""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_distribution": self.class_distribution,
            "entropy": self.calculate_entropy(),
            "gini": self.calculate_gini(),
            "samples_count": self.samples_count
        }

class FuzzyRuleNode(FuzzyTreeNode):
    """ファジィルールベースの特殊ノード"""
    
    def __init__(self, node_id: str, rules: List[Dict[str, Any]], depth: int = 0):
        super().__init__(node_id, depth)
        self.rules = rules                             # ファジィルールリスト
        self.rule_weights: Dict[int, float] = {}       # ルール重み
        self.aggregation_method = "weighted_average"   # 集約方法
        
    def add_rule(self, rule: Dict[str, Any], weight: float = 1.0) -> None:
        """ルールを追加"""
        rule_id = len(self.rules)
        self.rules.append(rule)
        self.rule_weights[rule_id] = weight
    
    def predict(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """ファジィルールベース予測"""
        
        activated_rules = []
        
        # 各ルールの発火度を計算
        for rule_id, rule in enumerate(self.rules):
            activation = self._evaluate_rule(rule, sample)
            if activation > 0:
                weight = self.rule_weights.get(rule_id, 1.0)
                activated_rules.append({
                    "rule_id": rule_id,
                    "activation": activation,
                    "weight": weight,
                    "consequent": rule.get("consequent", {})
                })
        
        # ルール集約
        if not activated_rules:
            return {"unknown": 1.0}  # デフォルト予測
        
        return self._aggregate_rules(activated_rules)
    
    def _evaluate_rule(self, rule: Dict[str, Any], sample: Dict[str, Any]) -> float:
        """ルールの発火度を評価"""
        
        antecedent = rule.get("antecedent", {})
        if not antecedent:
            return 0.0
        
        # 前件部の各条件を評価
        condition_values = []
        
        for feature, condition in antecedent.items():
            if feature not in sample:
                continue
                
            feature_value = sample[feature]
            linguistic_value = condition.get("linguistic_value", "medium")
            threshold = condition.get("threshold", 5.0)
            
            # メンバーシップ度を計算
            membership = self._calculate_membership(feature_value, linguistic_value, threshold)
            condition_values.append(membership)
        
        # 条件の結合（最小値：AND演算）
        return min(condition_values) if condition_values else 0.0
    
    def _calculate_membership(self, value: float, linguistic_value: str, threshold: float) -> float:
        """メンバーシップ度を計算"""
        
        if linguistic_value == "low":
            if value <= threshold - 1:
                return 1.0
            elif value <= threshold + 1:
                return max(0, 1.0 - (value - threshold + 1) / 2.0)
            else:
                return 0.0
                
        elif linguistic_value == "high":
            if value >= threshold + 1:
                return 1.0
            elif value >= threshold - 1:
                return max(0, (value - threshold + 1) / 2.0)
            else:
                return 0.0
                
        else:  # medium
            distance = abs(value - threshold)
            if distance <= 1:
                return 1.0 - distance
            else:
                return 0.0
    
    def _aggregate_rules(self, activated_rules: List[Dict[str, Any]]) -> Dict[str, float]:
        """ルール集約"""
        
        if self.aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(activated_rules)
        elif self.aggregation_method == "max_activation":
            return self._max_activation_aggregation(activated_rules)
        else:
            return self._weighted_average_aggregation(activated_rules)
    
    def _weighted_average_aggregation(self, activated_rules: List[Dict[str, Any]]) -> Dict[str, float]:
        """重み付き平均集約"""
        
        result = {}
        total_weight = 0.0
        
        for rule_info in activated_rules:
            activation = rule_info["activation"]
            weight = rule_info["weight"]
            consequent = rule_info["consequent"]
            
            effective_weight = activation * weight
            
            for class_name, class_value in consequent.items():
                if class_name not in result:
                    result[class_name] = 0.0
                result[class_name] += class_value * effective_weight
            
            total_weight += effective_weight
        
        # 正規化
        if total_weight > 0:
            for class_name in result:
                result[class_name] /= total_weight
        
        return result
    
    def _max_activation_aggregation(self, activated_rules: List[Dict[str, Any]]) -> Dict[str, float]:
        """最大発火度集約"""
        
        best_rule = max(activated_rules, key=lambda x: x["activation"])
        return best_rule["consequent"].copy()
    
    def is_leaf(self) -> bool:
        """ルールノードも葉として扱う"""
        return True
    
    def get_rule_info(self) -> Dict[str, Any]:
        """ルール情報を取得"""
        return {
            "rule_count": len(self.rules),
            "aggregation_method": self.aggregation_method,
            "active_rules": len([r for r in self.rules if r]),
            "total_weight": sum(self.rule_weights.values())
        }

class NodeFactory:
    """ノードファクトリクラス"""
    
    @staticmethod
    def create_internal_node(node_id: str, feature: str, threshold: float,
                           linguistic_value: str = "medium", depth: int = 0) -> FuzzyInternalNode:
        """内部ノードを作成"""
        split_condition = SplitCondition(
            feature=feature,
            threshold=threshold,
            linguistic_value=linguistic_value,
            membership_threshold=0.5
        )
        return FuzzyInternalNode(node_id, split_condition, depth)
    
    @staticmethod
    def create_leaf_node(node_id: str, class_distribution: Dict[str, float],
                        depth: int = 0, confidence: float = 1.0) -> FuzzyLeafNode:
        """葉ノードを作成"""
        return FuzzyLeafNode(node_id, class_distribution, depth, confidence)
    
    @staticmethod
    def create_rule_node(node_id: str, rules: List[Dict[str, Any]] = None,
                        depth: int = 0) -> FuzzyRuleNode:
        """ルールノードを作成"""
        if rules is None:
            rules = []
        return FuzzyRuleNode(node_id, rules, depth)

class NodeTraverser:
    """ノード探索ユーティリティ"""
    
    @staticmethod
    def traverse_preorder(root: FuzzyTreeNode, visit_func: callable) -> List[Any]:
        """前順探索"""
        results = []
        
        def _traverse(node):
            if node is not None:
                results.append(visit_func(node))
                if hasattr(node, 'children'):
                    for child in node.children.values():
                        _traverse(child)
        
        _traverse(root)
        return results
    
    @staticmethod
    def traverse_postorder(root: FuzzyTreeNode, visit_func: callable) -> List[Any]:
        """後順探索"""
        results = []
        
        def _traverse(node):
            if node is not None:
                if hasattr(node, 'children'):
                    for child in node.children.values():
                        _traverse(child)
                results.append(visit_func(node))
        
        _traverse(root)
        return results
    
    @staticmethod
    def find_leaves(root: FuzzyTreeNode) -> List[FuzzyTreeNode]:
        """すべての葉ノードを取得"""
        leaves = []
        
        def collect_leaf(node):
            if node.is_leaf():
                leaves.append(node)
            return None
        
        NodeTraverser.traverse_preorder(root, collect_leaf)
        return leaves
    
    @staticmethod
    def calculate_tree_depth(root: FuzzyTreeNode) -> int:
        """木の深さを計算"""
        max_depth = 0
        
        def update_depth(node):
            nonlocal max_depth
            max_depth = max(max_depth, node.depth)
            return None
        
        NodeTraverser.traverse_preorder(root, update_depth)
        return max_depth
    
    @staticmethod
    def count_nodes(root: FuzzyTreeNode) -> Dict[str, int]:
        """ノード数を計算"""
        counts = {"total": 0, "internal": 0, "leaf": 0, "rule": 0}
        
        def count_node(node):
            counts["total"] += 1
            if isinstance(node, FuzzyInternalNode):
                counts["internal"] += 1
            elif isinstance(node, FuzzyLeafNode):
                counts["leaf"] += 1
            elif isinstance(node, FuzzyRuleNode):
                counts["rule"] += 1
            return None
        
        NodeTraverser.traverse_preorder(root, count_node)
        return counts