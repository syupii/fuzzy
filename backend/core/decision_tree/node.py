# core/decision_tree/node.py - ファジィ決定木ノード（完全版）

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import json
import logging

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
        self.creation_time = 0.0
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
                
                total_weight += membership
                
                # パスの統合
                final_prediction["path"].extend(child_prediction.get("path", []))
        
        # 正規化
        if total_weight > 0:
            for class_name in weighted_probabilities:
                weighted_probabilities[class_name] /= total_weight
        
        # 最終予測クラスの決定
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
        """特徴値に対する各分岐への帰属度を計算"""
        
        # 簡易版：閾値ベースの二分岐
        if feature_value >= self.split_condition.threshold:
            return {"left": 1.0, "right": 0.0}
        else:
            return {"left": 0.0, "right": 1.0}
    
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return False
    
    def get_child_count(self) -> int:
        """子ノード数を取得"""
        return len(self.children)
    
    def get_split_info(self) -> Dict[str, Any]:
        """分岐情報を取得"""
        return {
            "feature": self.split_condition.feature,
            "threshold": self.split_condition.threshold,
            "linguistic_value": self.split_condition.linguistic_value,
            "usage_count": self.split_usage_count,
            "branch_statistics": self.branch_statistics
        }
    
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
            "split_usage_count": self.split_usage_count,
            "branch_statistics": self.branch_statistics,
            "metadata": self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        self.node_id = data.get("node_id", self.node_id)
        self.depth = data.get("depth", self.depth)
        self.samples_count = data.get("samples_count", 0)
        self.purity = data.get("purity", 0.0)
        self.split_usage_count = data.get("split_usage_count", 0)
        self.branch_statistics = data.get("branch_statistics", {})
        self.metadata = data.get("metadata", {})
        
        # 分岐条件の復元
        split_data = data.get("split_condition", {})
        self.split_condition = SplitCondition(
            feature=split_data.get("feature", ""),
            threshold=split_data.get("threshold", 0.0),
            linguistic_value=split_data.get("linguistic_value", ""),
            membership_threshold=split_data.get("membership_threshold", 0.1),
            operator=split_data.get("operator", ">")
        )

class FuzzyLeafNode(FuzzyTreeNode):
    """ファジィ決定木の葉ノード"""
    
    def __init__(self, node_id: str, predicted_class: str, 
                 class_probabilities: Dict[str, float], 
                 confidence: float, depth: int = 0):
        super().__init__(node_id, depth)
        self.predicted_class = predicted_class
        self.class_probabilities = class_probabilities
        self.confidence = confidence
        
        # 葉ノード固有の統計
        self.prediction_accuracy = 0.0
        self.class_distribution = class_probabilities.copy()
        
    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """予測を実行"""
        
        prediction = {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities.copy(),
            "path": [self.node_id],
            "leaf_node": True,
            "samples_count": self.samples_count
        }
        
        # 統計更新
        self.update_prediction_stats(self.confidence)
        
        return prediction
    
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return True
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """予測サマリーを取得"""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_distribution": self.class_probabilities,
            "prediction_count": self.prediction_count,
            "average_confidence": np.mean(self.confidence_history) if self.confidence_history else 0.0
        }
    
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
            "prediction_count": self.prediction_count,
            "prediction_accuracy": self.prediction_accuracy,
            "metadata": self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        self.node_id = data.get("node_id", self.node_id)
        self.depth = data.get("depth", self.depth)
        self.predicted_class = data.get("predicted_class", "unknown")
        self.class_probabilities = data.get("class_probabilities", {})
        self.confidence = data.get("confidence", 0.0)
        self.samples_count = data.get("samples_count", 0)
        self.purity = data.get("purity", 0.0)
        self.prediction_count = data.get("prediction_count", 0)
        self.prediction_accuracy = data.get("prediction_accuracy", 0.0)
        self.metadata = data.get("metadata", {})

class FuzzyRuleNode(FuzzyTreeNode):
    """ファジィルール型ノード"""
    
    def __init__(self, node_id: str, rule_conditions: List[str], 
                 conclusion: str, confidence: float, depth: int = 0):
        super().__init__(node_id, depth)
        self.rule_conditions = rule_conditions
        self.conclusion = conclusion
        self.confidence = confidence
        
        # ルール関連統計
        self.activation_count = 0
        self.activation_strength_history: List[float] = []
        
    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """ルールベース予測"""
        
        # ルール条件の評価
        activation_strength = self._evaluate_rule_conditions(sample)
        
        prediction = {
            "predicted_class": self.conclusion,
            "confidence": self.confidence * activation_strength,
            "class_probabilities": {self.conclusion: self.confidence * activation_strength},
            "path": [self.node_id],
            "rule_node": True,
            "activation_strength": activation_strength,
            "rule_conditions": self.rule_conditions
        }
        
        # 統計更新
        if activation_strength > 0:
            self.activation_count += 1
            self.activation_strength_history.append(activation_strength)
            
        self.update_prediction_stats(prediction["confidence"])
        
        return prediction
    
    def _evaluate_rule_conditions(self, sample: Dict[str, Any]) -> float:
        """ルール条件の評価"""
        
        # 簡易版：全条件が満たされた場合の強度を返す
        # 実際の実装では、ファジィルールエンジンを使用
        
        total_strength = 1.0
        evaluated_conditions = 0
        
        for condition in self.rule_conditions:
            # 条件文字列の簡易解析
            strength = self._evaluate_single_condition(condition, sample)
            total_strength = min(total_strength, strength)  # AND結合
            evaluated_conditions += 1
        
        return total_strength if evaluated_conditions > 0 else 0.0
    
    def _evaluate_single_condition(self, condition: str, sample: Dict[str, Any]) -> float:
        """単一条件の評価"""
        
        # 簡易版の条件評価
        # 実際の実装では、より詳細な解析が必要
        
        # "feature IS linguistic_value" の形式を仮定
        parts = condition.split()
        if len(parts) >= 3:
            feature_name = parts[0]
            linguistic_value = parts[2] if len(parts) > 2 else "unknown"
            
            feature_value = sample.get(feature_name, 0.0)
            
            # 言語値に基づく簡易評価
            if linguistic_value == "high":
                return max(0.0, (feature_value - 7.0) / 3.0)
            elif linguistic_value == "medium":
                return max(0.0, 1.0 - abs(feature_value - 5.0) / 2.5)
            elif linguistic_value == "low":
                return max(0.0, (3.0 - feature_value) / 3.0)
        
        return 0.5  # デフォルト値
    
    def is_leaf(self) -> bool:
        """葉ノードかどうか"""
        return True
    
    def get_rule_info(self) -> Dict[str, Any]:
        """ルール情報を取得"""
        return {
            "conditions": self.rule_conditions,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "activation_count": self.activation_count,
            "average_activation_strength": (
                np.mean(self.activation_strength_history) 
                if self.activation_strength_history else 0.0
            )
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "node_type": "rule",
            "node_id": self.node_id,
            "depth": self.depth,
            "rule_conditions": self.rule_conditions,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "activation_count": self.activation_count,
            "samples_count": self.samples_count,
            "metadata": self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        self.node_id = data.get("node_id", self.node_id)
        self.depth = data.get("depth", self.depth)
        self.rule_conditions = data.get("rule_conditions", [])
        self.conclusion = data.get("conclusion", "unknown")
        self.confidence = data.get("confidence", 0.0)
        self.activation_count = data.get("activation_count", 0)
        self.samples_count = data.get("samples_count", 0)
        self.metadata = data.get("metadata", {})

class NodeTraverser:
    """ノード探索ユーティリティ"""
    
    @staticmethod
    def get_all_nodes(root: FuzzyTreeNode) -> List[FuzzyTreeNode]:
        """全ノードを取得"""
        
        nodes = []
        
        def traverse(node: FuzzyTreeNode):
            nodes.append(node)
            if not node.is_leaf() and hasattr(node, 'children'):
                for child in node.children.values():
                    traverse(child)
        
        traverse(root)
        return nodes
    
    @staticmethod
    def get_leaf_nodes(root: FuzzyTreeNode) -> List[FuzzyTreeNode]:
        """葉ノードのみを取得"""
        
        all_nodes = NodeTraverser.get_all_nodes(root)
        return [node for node in all_nodes if node.is_leaf()]
    
    @staticmethod
    def get_internal_nodes(root: FuzzyTreeNode) -> List[FuzzyTreeNode]:
        """内部ノードのみを取得"""
        
        all_nodes = NodeTraverser.get_all_nodes(root)
        return [node for node in all_nodes if not node.is_leaf()]
    
    @staticmethod
    def get_nodes_at_depth(root: FuzzyTreeNode, target_depth: int) -> List[FuzzyTreeNode]:
        """指定深度のノードを取得"""
        
        all_nodes = NodeTraverser.get_all_nodes(root)
        return [node for node in all_nodes if node.depth == target_depth]
    
    @staticmethod
    def get_tree_statistics(root: FuzzyTreeNode) -> Dict[str, Any]:
        """木統計を取得"""
        
        all_nodes = NodeTraverser.get_all_nodes(root)
        leaf_nodes = [node for node in all_nodes if node.is_leaf()]
        internal_nodes = [node for node in all_nodes if not node.is_leaf()]
        
        max_depth = max(node.depth for node in all_nodes) if all_nodes else 0
        
        return {
            "total_nodes": len(all_nodes),
            "leaf_nodes": len(leaf_nodes),
            "internal_nodes": len(internal_nodes),
            "max_depth": max_depth,
            "average_depth": np.mean([node.depth for node in leaf_nodes]) if leaf_nodes else 0,
            "total_predictions": sum(node.prediction_count for node in all_nodes),
            "tree_structure": NodeTraverser._build_structure_info(root)
        }
    
    @staticmethod
    def _build_structure_info(node: FuzzyTreeNode) -> Dict[str, Any]:
        """木構造情報を構築"""
        
        info = {
            "node_id": node.node_id,
            "node_type": "leaf" if node.is_leaf() else "internal",
            "depth": node.depth,
            "prediction_count": node.prediction_count
        }
        
        if not node.is_leaf() and hasattr(node, 'children'):
            info["children"] = {}
            for branch_name, child in node.children.items():
                info["children"][branch_name] = NodeTraverser._build_structure_info(child)
        
        return info

# 使用例とテスト
def test_fuzzy_tree_nodes():
    """ファジィ決定木ノードのテスト"""
    
    print("🌳 ファジィ決定木ノードテスト開始")
    
    # 葉ノードの作成
    leaf = FuzzyLeafNode(
        node_id="leaf_1",
        predicted_class="high_match",
        class_probabilities={"high_match": 0.8, "medium_match": 0.2},
        confidence=0.8,
        depth=2
    )
    
    # 内部ノードの作成
    split_condition = SplitCondition(
        feature="research_intensity",
        threshold=7.0,
        linguistic_value="high",
        membership_threshold=0.2
    )
    
    internal = FuzzyInternalNode(
        node_id="internal_1",
        split_condition=split_condition,
        depth=1
    )
    
    internal.add_child("left", leaf)
    
    # ルールノードの作成
    rule = FuzzyRuleNode(
        node_id="rule_1",
        rule_conditions=["research_intensity IS high", "advisor_style IS high"],
        conclusion="high_match",
        confidence=0.9,
        depth=2
    )
    
    internal.add_child("right", rule)
    
    # テストサンプル
    test_sample = {
        "research_intensity": 8.5,
        "advisor_style": 7.0,
        "team_work": 6.0
    }
    
    # 予測テスト
    prediction = internal.predict(test_sample)
    
    print(f"📊 予測結果:")
    print(f"  予測クラス: {prediction['predicted_class']}")
    print(f"  信頼度: {prediction['confidence']:.3f}")
    print(f"  クラス確率: {prediction['class_probabilities']}")
    
    # 木統計
    stats = NodeTraverser.get_tree_statistics(internal)
    print(f"\n📈 木統計:")
    print(f"  総ノード数: {stats['total_nodes']}")
    print(f"  葉ノード数: {stats['leaf_nodes']}")
    print(f"  最大深度: {stats['max_depth']}")
    
    # 辞書変換テスト
    tree_dict = internal.to_dict()
    print(f"\n💾 辞書変換成功: {len(json.dumps(tree_dict))}文字")
    
    print("✅ ファジィ決定木ノードテスト完了")

if __name__ == "__main__":
    test_fuzzy_tree_nodes()