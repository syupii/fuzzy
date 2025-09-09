# core/decision_tree/node.py - ファジィ決定木ノード（完全版）

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import json
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class SplitCondition:
    """分岐条件"""
    feature: str                    # 分岐に使用する特徴量
    threshold: float                # 閾値
    linguistic_value: str           # 言語値（"low", "medium", "high"など）
    membership_threshold: float     # メンバーシップ関数の閾値
    operator: str = ">"            # 比較演算子

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "linguistic_value": self.linguistic_value,
            "membership_threshold": self.membership_threshold,
            "operator": self.operator
        }

    def __str__(self) -> str:
        return f"{self.feature} IS {self.linguistic_value} (threshold: {self.threshold:.3f})"

class FuzzyTreeNode(ABC):
    """ファジィ決定木ノードの抽象基底クラス"""
    
    def __init__(self, node_id: str, depth: int = 0):
        self.node_id = node_id
        self.depth = depth
        self.parent: Optional['FuzzyTreeNode'] = None
        self.samples_count = 0
        self.purity = 0.0
        
        # ノード統計
        self.creation_time = time.time()
        self.last_prediction_time = 0.0
        self.prediction_count = 0
        self.confidence_history: List[float] = []
        
        # メタデータ
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """予測を実行"""
        pass
    
    @abstractmethod
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        pass
    
    def get_path_to_root(self) -> List['FuzzyTreeNode']:
        """ルートまでのパスを取得"""
        path = [self]
        current = self.parent
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_depth(self) -> int:
        """深度を取得"""
        return self.depth
    
    def update_prediction_stats(self, confidence: float):
        """予測統計を更新"""
        self.prediction_count += 1
        self.confidence_history.append(confidence)
        self.last_prediction_time = time.time()
        
        # 最新100回の履歴のみ保持
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]

class FuzzyInternalNode(FuzzyTreeNode):
    """ファジィ決定木の内部ノード"""
    
    def __init__(self, node_id: str, split_condition: SplitCondition, depth: int = 0):
        super().__init__(node_id, depth)
        self.split_condition = split_condition
        self.children: Dict[str, FuzzyTreeNode] = {}  # 子ノード（"left", "right"など）
        self.split_values: Dict[str, float] = {}      # 各分岐の値
        
        # 分岐統計
        self.split_usage_count = 0
        self.branch_statistics: Dict[str, Dict[str, Any]] = {}
        
    def add_child(self, branch_name: str, child_node: 'FuzzyTreeNode') -> None:
        """子ノードを追加"""
        self.children[branch_name] = child_node
        child_node.parent = self
        
        # 分岐統計の初期化
        self.branch_statistics[branch_name] = {
            "usage_count": 0,
            "average_confidence": 0.0,
            "prediction_count": 0
        }
    
    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """予測を実行（ファジィ推論）"""
        
        self.split_usage_count += 1
        feature_value = sample.get(self.split_condition.feature, 0.0)
        
        # 各子ノードへの帰属度を計算
        membership_values = self._calculate_memberships(feature_value)
        
        # 各子ノードの予測を重み付き平均
        final_prediction = {
            "predicted_class": None,
            "confidence": 0.0,
            "class_probabilities": {},
            "path": [self.node_id],
            "membership_values": membership_values,
            "split_feature": self.split_condition.feature,
            "split_value": feature_value
        }
        
        total_weight = 0.0
        weighted_probabilities = {}
        
        for branch_name, membership in membership_values.items():
            if membership > 0 and branch_name in self.children:
                child_prediction = self.children[branch_name].predict(sample)
                
                # 分岐統計の更新
                self.branch_statistics[branch_name]["usage_count"] += 1
                self.branch_statistics[branch_name]["prediction_count"] += 1
                
                # 重み付きで統合
                for class_name, prob in child_prediction.get("class_probabilities", {}).items():
                    if class_name not in weighted_probabilities:
                        weighted_probabilities[class_name] = 0.0
                    weighted_probabilities[class_name] += prob * membership
                
                # パスの更新
                if child_prediction.get("path"):
                    final_prediction["path"].extend(child_prediction["path"])
                
                total_weight += membership
        
        # 正規化
        if total_weight > 0:
            for class_name in weighted_probabilities:
                weighted_probabilities[class_name] /= total_weight
        
        # 最も確率の高いクラスを選択
        if weighted_probabilities:
            predicted_class = max(weighted_probabilities, key=weighted_probabilities.get)
            confidence = weighted_probabilities[predicted_class]
        else:
            predicted_class = "unknown"
            confidence = 0.0
        
        final_prediction["predicted_class"] = predicted_class
        final_prediction["confidence"] = confidence
        final_prediction["class_probabilities"] = weighted_probabilities
        
        # 統計更新
        self.update_prediction_stats(confidence)
        
        return final_prediction
    
    def _calculate_memberships(self, feature_value: float) -> Dict[str, float]:
        """特徴値に対する各分岐のメンバーシップ度を計算"""
        
        memberships = {}
        
        # 簡易実装：閾値による二分岐
        if self.split_condition.operator == ">":
            # 高い値への帰属度
            high_membership = 1.0 if feature_value > self.split_condition.threshold else 0.0
            low_membership = 1.0 - high_membership
            
            memberships["high"] = high_membership
            memberships["low"] = low_membership
        else:
            # デフォルト：均等分割
            memberships["left"] = 0.5
            memberships["right"] = 0.5
        
        return memberships
    
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "node_type": "internal",
            "node_id": self.node_id,
            "depth": self.depth,
            "split_condition": self.split_condition.to_dict(),
            "children": {name: child.to_dict() for name, child in self.children.items()},
            "samples_count": self.samples_count,
            "purity": self.purity,
            "prediction_count": self.prediction_count,
            "metadata": self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        self.node_id = data["node_id"]
        self.depth = data["depth"]
        self.samples_count = data.get("samples_count", 0)
        self.purity = data.get("purity", 0.0)
        self.prediction_count = data.get("prediction_count", 0)
        self.metadata = data.get("metadata", {})
        
        # 分岐条件の復元
        split_data = data["split_condition"]
        self.split_condition = SplitCondition(
            feature=split_data["feature"],
            threshold=split_data["threshold"],
            linguistic_value=split_data["linguistic_value"],
            membership_threshold=split_data["membership_threshold"],
            operator=split_data.get("operator", ">")
        )

class FuzzyLeafNode(FuzzyTreeNode):
    """ファジィ決定木の葉ノード"""
    
    def __init__(self, node_id: str, predicted_class: str, 
                 class_probabilities: Dict[str, float] = None, depth: int = 0):
        super().__init__(node_id, depth)
        self.predicted_class = predicted_class
        self.class_probabilities = class_probabilities or {predicted_class: 1.0}
        self.confidence = max(self.class_probabilities.values()) if self.class_probabilities else 0.0
        
        # 葉ノード特有の統計
        self.leaf_purity = self.confidence
        self.misclassification_count = 0
        
    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """予測を実行"""
        
        prediction = {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities.copy(),
            "path": [self.node_id],
            "leaf_node": True,
            "samples_count": self.samples_count,
            "leaf_purity": self.leaf_purity
        }
        
        # 統計更新
        self.update_prediction_stats(self.confidence)
        
        return prediction
    
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return True
    
    def update_class_distribution(self, new_class: str, weight: float = 1.0):
        """クラス分布を更新"""
        
        if new_class not in self.class_probabilities:
            self.class_probabilities[new_class] = 0.0
        
        # 重み付き更新
        total_weight = sum(self.class_probabilities.values()) + weight
        for class_name in self.class_probabilities:
            self.class_probabilities[class_name] = self.class_probabilities[class_name] / total_weight
        
        self.class_probabilities[new_class] += weight / total_weight
        
        # 予測クラスと信頼度の更新
        self.predicted_class = max(self.class_probabilities, key=self.class_probabilities.get)
        self.confidence = self.class_probabilities[self.predicted_class]
        self.leaf_purity = self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "node_type": "leaf",
            "node_id": self.node_id,
            "depth": self.depth,
            "predicted_class": self.predicted_class,
            "class_probabilities": self.class_probabilities,
            "confidence": self.confidence,
            "samples_count": self.samples_count,
            "purity": self.purity,
            "leaf_purity": self.leaf_purity,
            "prediction_count": self.prediction_count,
            "metadata": self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        self.node_id = data["node_id"]
        self.depth = data["depth"]
        self.predicted_class = data["predicted_class"]
        self.class_probabilities = data["class_probabilities"]
        self.confidence = data["confidence"]
        self.samples_count = data.get("samples_count", 0)
        self.purity = data.get("purity", 0.0)
        self.leaf_purity = data.get("leaf_purity", self.confidence)
        self.prediction_count = data.get("prediction_count", 0)
        self.metadata = data.get("metadata", {})

class FuzzyRuleNode(FuzzyTreeNode):
    """ファジィルールノード（特殊ノード）"""
    
    def __init__(self, node_id: str, rule_conditions: List[str], 
                 rule_conclusion: str, rule_weight: float = 1.0, depth: int = 0):
        super().__init__(node_id, depth)
        self.rule_conditions = rule_conditions
        self.rule_conclusion = rule_conclusion
        self.rule_weight = rule_weight
        
        # ルール統計
        self.activation_count = 0
        self.total_activation = 0.0
        self.average_activation = 0.0
        
    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """ルールベース予測"""
        
        # ルール条件の評価（簡易実装）
        activation = self._evaluate_rule_conditions(sample)
        
        prediction = {
            "predicted_class": self.rule_conclusion,
            "confidence": activation * self.rule_weight,
            "class_probabilities": {self.rule_conclusion: activation * self.rule_weight},
            "path": [self.node_id],
            "rule_activation": activation,
            "rule_weight": self.rule_weight,
            "rule_conditions": self.rule_conditions
        }
        
        # ルール統計更新
        self.activation_count += 1
        self.total_activation += activation
        self.average_activation = self.total_activation / self.activation_count
        
        # 統計更新
        self.update_prediction_stats(activation * self.rule_weight)
        
        return prediction
    
    def _evaluate_rule_conditions(self, sample: Dict[str, Any]) -> float:
        """ルール条件の評価"""
        
        if not self.rule_conditions:
            return 1.0
        
        # 簡易実装：ランダムな活性化度
        # 実際の実装では、ファジィルール評価エンジンを使用
        activation = 1.0
        
        for condition in self.rule_conditions:
            # 条件の解析と評価（簡易版）
            if "high" in condition.lower():
                activation *= 0.8
            elif "medium" in condition.lower():
                activation *= 0.6
            elif "low" in condition.lower():
                activation *= 0.4
            else:
                activation *= 0.5
        
        return min(1.0, max(0.0, activation))
    
    def is_leaf(self) -> bool:
        """葉ノードかどうか（ルールノードは葉として扱う）"""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "node_type": "rule",
            "node_id": self.node_id,
            "depth": self.depth,
            "rule_conditions": self.rule_conditions,
            "rule_conclusion": self.rule_conclusion,
            "rule_weight": self.rule_weight,
            "activation_count": self.activation_count,
            "average_activation": self.average_activation,
            "samples_count": self.samples_count,
            "purity": self.purity,
            "prediction_count": self.prediction_count,
            "metadata": self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        self.node_id = data["node_id"]
        self.depth = data["depth"]
        self.rule_conditions = data["rule_conditions"]
        self.rule_conclusion = data["rule_conclusion"]
        self.rule_weight = data["rule_weight"]
        self.activation_count = data.get("activation_count", 0)
        self.average_activation = data.get("average_activation", 0.0)
        self.samples_count = data.get("samples_count", 0)
        self.purity = data.get("purity", 0.0)
        self.prediction_count = data.get("prediction_count", 0)
        self.metadata = data.get("metadata", {})

class NodeTraverser:
    """ノード走査ユーティリティ"""
    
    @staticmethod
    def depth_first_search(root: FuzzyTreeNode, 
                          visit_func: callable = None) -> List[FuzzyTreeNode]:
        """深度優先探索"""
        
        visited = []
        
        def dfs(node: FuzzyTreeNode):
            visited.append(node)
            
            if visit_func:
                visit_func(node)
            
            if hasattr(node, 'children'):
                for child in node.children.values():
                    dfs(child)
        
        dfs(root)
        return visited
    
    @staticmethod
    def breadth_first_search(root: FuzzyTreeNode, 
                           visit_func: callable = None) -> List[FuzzyTreeNode]:
        """幅優先探索"""
        
        visited = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            visited.append(node)
            
            if visit_func:
                visit_func(node)
            
            if hasattr(node, 'children'):
                queue.extend(node.children.values())
        
        return visited
    
    @staticmethod
    def find_leaves(root: FuzzyTreeNode) -> List[FuzzyTreeNode]:
        """葉ノードを検索"""
        
        leaves = []
        
        def collect_leaves(node: FuzzyTreeNode):
            if node.is_leaf():
                leaves.append(node)
        
        NodeTraverser.depth_first_search(root, collect_leaves)
        return leaves
    
    @staticmethod
    def calculate_tree_depth(root: FuzzyTreeNode) -> int:
        """木の深度を計算"""
        
        max_depth = 0
        
        def check_depth(node: FuzzyTreeNode):
            nonlocal max_depth
            max_depth = max(max_depth, node.depth)
        
        NodeTraverser.depth_first_search(root, check_depth)
        return max_depth
    
    @staticmethod
    def count_nodes_by_type(root: FuzzyTreeNode) -> Dict[str, int]:
        """ノードタイプ別カウント"""
        
        counts = {"internal": 0, "leaf": 0, "rule": 0}
        
        def count_node(node: FuzzyTreeNode):
            if isinstance(node, FuzzyInternalNode):
                counts["internal"] += 1
            elif isinstance(node, FuzzyRuleNode):
                counts["rule"] += 1
            elif isinstance(node, FuzzyLeafNode):
                counts["leaf"] += 1
        
        NodeTraverser.depth_first_search(root, count_node)
        return counts

# 使用例とテスト
def test_fuzzy_nodes():
    """ファジィノードのテスト"""
    
    print("🌳 ファジィ決定木ノードテスト開始")
    
    # 分岐条件の作成
    split_condition = SplitCondition(
        feature="research_intensity",
        threshold=5.0,
        linguistic_value="medium",
        membership_threshold=0.5
    )
    
    # 内部ノードの作成
    internal_node = FuzzyInternalNode("root", split_condition, 0)
    
    # 葉ノードの作成
    leaf_high = FuzzyLeafNode("leaf_high", "high_match", {"high_match": 0.8, "medium_match": 0.2}, 1)
    leaf_low = FuzzyLeafNode("leaf_low", "low_match", {"low_match": 0.9, "medium_match": 0.1}, 1)
    
    # 子ノードの追加
    internal_node.add_child("high", leaf_high)
    internal_node.add_child("low", leaf_low)
    
    # 予測テスト
    test_sample = {"research_intensity": 7.0, "advisor_style": 6.0}
    
    print("\n📊 予測テスト:")
    prediction = internal_node.predict(test_sample)
    print(f"  予測クラス: {prediction['predicted_class']}")
    print(f"  信頼度: {prediction['confidence']:.3f}")
    print(f"  経路: {' -> '.join(prediction['path'])}")
    
    # 木構造の解析
    print("\n🔍 木構造解析:")
    nodes = NodeTraverser.depth_first_search(internal_node)
    print(f"  総ノード数: {len(nodes)}")
    
    node_counts = NodeTraverser.count_nodes_by_type(internal_node)
    print(f"  内部ノード: {node_counts['internal']}")
    print(f"  葉ノード: {node_counts['leaf']}")
    print(f"  ルールノード: {node_counts['rule']}")
    
    max_depth = NodeTraverser.calculate_tree_depth(internal_node)
    print(f"  最大深度: {max_depth}")
    
    print("✅ ファジィ決定木ノードテスト完了")

if __name__ == "__main__":
    test_fuzzy_nodes()