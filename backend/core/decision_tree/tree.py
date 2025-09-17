# core/decision_tree/tree.py - ファジィ決定木システム

import numpy as np
import math
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class FuzzyCondition:
    """ファジィ条件"""
    attribute: str
    linguistic_value: str  # "low", "medium", "high"
    membership_degree: float
    threshold: float = 0.0

@dataclass
class FuzzyRule:
    """ファジィルール"""
    conditions: List[FuzzyCondition]
    conclusion: str
    confidence: float
    support: float
    rule_id: int = 0

class FuzzyTreeNode(ABC):
    """ファジィ決定木ノードの抽象基底クラス"""
    
    def __init__(self, node_id: str, depth: int = 0):
        self.node_id = node_id
        self.depth = depth
        self.samples_count = 0
        self.class_distribution: Dict[str, float] = {}
        
    @abstractmethod
    def predict(self, instance: Dict[str, float]) -> Dict[str, float]:
        """予測実行"""
        pass

class FuzzyLeafNode(FuzzyTreeNode):
    """ファジィ葉ノード"""
    
    def __init__(self, node_id: str, predicted_class: str, 
                 confidence: float, depth: int = 0):
        super().__init__(node_id, depth)
        self.predicted_class = predicted_class
        self.confidence = confidence
        
    def predict(self, instance: Dict[str, float]) -> Dict[str, float]:
        """葉ノードでの予測"""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "node_id": self.node_id,
            "depth": self.depth
        }

class FuzzyInternalNode(FuzzyTreeNode):
    """ファジィ内部ノード"""
    
    def __init__(self, node_id: str, attribute: str, depth: int = 0):
        super().__init__(node_id, depth)
        self.attribute = attribute
        self.children: Dict[str, FuzzyTreeNode] = {}  # linguistic_value -> child_node
        self.membership_functions: Dict[str, Tuple[float, float, float]] = {}
        
    def add_child(self, linguistic_value: str, child_node: FuzzyTreeNode):
        """子ノードを追加"""
        self.children[linguistic_value] = child_node
        
    def set_membership_function(self, linguistic_value: str, mf_params: Tuple[float, float, float]):
        """メンバーシップ関数を設定（三角形関数）"""
        self.membership_functions[linguistic_value] = mf_params
    
    def calculate_membership(self, value: float, linguistic_value: str) -> float:
        """メンバーシップ度を計算"""
        if linguistic_value not in self.membership_functions:
            return 0.0
            
        a, b, c = self.membership_functions[linguistic_value]
        
        if value <= a or value >= c:
            return 0.0
        elif value == b:
            return 1.0
        elif value < b:
            return (value - a) / (b - a)
        else:
            return (c - value) / (c - b)
    
    def predict(self, instance: Dict[str, float]) -> Dict[str, float]:
        """内部ノードでの予測"""
        if self.attribute not in instance:
            # デフォルト予測
            return {"predicted_class": "medium", "confidence": 0.5, 
                   "node_id": self.node_id, "depth": self.depth}
        
        attribute_value = instance[self.attribute]
        predictions = {}
        max_membership = 0.0
        best_prediction = None
        
        # 各子ノードでの予測を重み付き統合
        for linguistic_value, child_node in self.children.items():
            membership = self.calculate_membership(attribute_value, linguistic_value)
            
            if membership > 0.1:  # 閾値以上のメンバーシップのみ考慮
                child_prediction = child_node.predict(instance)
                
                # 重み付き予測
                weighted_confidence = child_prediction["confidence"] * membership
                predicted_class = child_prediction["predicted_class"]
                
                if predicted_class in predictions:
                    predictions[predicted_class] += weighted_confidence
                else:
                    predictions[predicted_class] = weighted_confidence
                
                # 最大メンバーシップの子ノードを記録
                if membership > max_membership:
                    max_membership = membership
                    best_prediction = child_prediction
        
        # 予測結果を正規化
        if predictions:
            total_weight = sum(predictions.values())
            normalized_predictions = {k: v/total_weight for k, v in predictions.items()}
            
            # 最も高い確信度のクラスを選択
            best_class = max(normalized_predictions, key=normalized_predictions.get)
            confidence = normalized_predictions[best_class]
            
            return {
                "predicted_class": best_class,
                "confidence": confidence,
                "class_probabilities": normalized_predictions,
                "node_id": self.node_id,
                "depth": self.depth,
                "membership_degree": max_membership
            }
        
        # フォールバック
        return best_prediction if best_prediction else {
            "predicted_class": "medium", "confidence": 0.5, 
            "node_id": self.node_id, "depth": self.depth
        }

class FuzzyDecisionTree:
    """ファジィ決定木クラス"""
    
    def __init__(self, tree_id: str = None, max_depth: int = 6):
        self.tree_id = tree_id or f"fuzzy_tree_{int(time.time())}"
        self.max_depth = max_depth
        self.root: Optional[FuzzyTreeNode] = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = ["very_low", "low", "medium", "high", "very_high"]
        
        # 学習用パラメータ
        self.min_samples_split = 5
        self.min_samples_leaf = 2
        self.min_impurity_decrease = 0.01
        
        # 統計情報
        self.node_count = 0
        self.training_samples = 0
        self.training_accuracy = 0.0
        self.rules: List[FuzzyRule] = []
        
    def fit(self, X: List[Dict[str, float]], y: List[str]):
        """決定木を学習"""
        logger.info(f"ファジィ決定木学習開始: サンプル数 {len(X)}")
        
        self.training_samples = len(X)
        self.feature_names = list(X[0].keys()) if X else []
        
        # ルートノードから再帰的に構築
        self.root = self._build_tree(X, y, depth=0, node_id="root")
        
        # ルール抽出
        self._extract_rules()
        
        # 学習精度の計算
        self.training_accuracy = self._calculate_accuracy(X, y)
        
        logger.info(f"決定木構築完了: ノード数 {self.node_count}, 精度 {self.training_accuracy:.3f}")
    
    def _build_tree(self, X: List[Dict[str, float]], y: List[str], 
                   depth: int, node_id: str) -> FuzzyTreeNode:
        """再帰的に決定木を構築"""
        
        self.node_count += 1
        
        # 停止条件のチェック
        if (depth >= self.max_depth or 
            len(X) < self.min_samples_split or
            len(set(y)) == 1):
            
            # 葉ノードを作成
            most_common_class = Counter(y).most_common(1)[0][0]
            confidence = y.count(most_common_class) / len(y)
            return FuzzyLeafNode(node_id, most_common_class, confidence, depth)
        
        # 最良の分割を選択
        best_split = self._find_best_split(X, y)
        
        if best_split is None:
            # 分割できない場合は葉ノードを作成
            most_common_class = Counter(y).most_common(1)[0][0]
            confidence = y.count(most_common_class) / len(y)
            return FuzzyLeafNode(node_id, most_common_class, confidence, depth)
        
        # 内部ノードを作成
        attribute, splits = best_split
        internal_node = FuzzyInternalNode(node_id, attribute, depth)
        
        # メンバーシップ関数を設定
        values = [sample[attribute] for sample in X]
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val
        
        # 3つの言語値（low, medium, high）のメンバーシップ関数
        internal_node.set_membership_function("low", 
            (min_val, min_val, min_val + 0.4 * range_val))
        internal_node.set_membership_function("medium", 
            (min_val + 0.2 * range_val, min_val + 0.5 * range_val, min_val + 0.8 * range_val))
        internal_node.set_membership_function("high", 
            (min_val + 0.6 * range_val, max_val, max_val))
        
        # 各分割に対して子ノードを再帰的に構築
        for linguistic_value, (X_subset, y_subset) in splits.items():
            if len(X_subset) >= self.min_samples_leaf:
                child_id = f"{node_id}_{linguistic_value}"
                child_node = self._build_tree(X_subset, y_subset, depth + 1, child_id)
                internal_node.add_child(linguistic_value, child_node)
        
        return internal_node
    
    def _find_best_split(self, X: List[Dict[str, float]], y: List[str]) -> Optional[Tuple[str, Dict[str, Tuple[List[Dict[str, float]], List[str]]]]]:
        """最良の分割を見つける"""
        
        if len(set(y)) <= 1:
            return None
        
        best_gain = -1
        best_split = None
        current_impurity = self._calculate_impurity(y)
        
        # 各属性について最良の分割を探索
        for attribute in self.feature_names:
            split_result = self._evaluate_fuzzy_split(X, y, attribute)
            
            if split_result is None:
                continue
                
            gain = split_result[0]
            splits = split_result[1]
            
            if gain > best_gain and gain > self.min_impurity_decrease:
                best_gain = gain
                best_split = (attribute, splits)
        
        return best_split
    
    def _evaluate_fuzzy_split(self, X: List[Dict[str, float]], y: List[str], 
                             attribute: str) -> Optional[Tuple[float, Dict[str, Tuple[List[Dict[str, float]], List[str]]]]]:
        """ファジィ分割の評価"""
        
        # 属性値の範囲を取得
        values = [sample[attribute] for sample in X]
        if len(set(values)) <= 1:
            return None
            
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val
        
        # ファジィ分割の定義
        splits = {"low": ([], []), "medium": ([], []), "high": ([], [])}
        
        # 各サンプルを最も適合する言語値に分類
        for i, sample in enumerate(X):
            value = sample[attribute]
            
            # メンバーシップ度を計算
            low_membership = self._triangular_membership(value, min_val, min_val, min_val + 0.4 * range_val)
            medium_membership = self._triangular_membership(value, min_val + 0.2 * range_val, min_val + 0.5 * range_val, min_val + 0.8 * range_val)
            high_membership = self._triangular_membership(value, min_val + 0.6 * range_val, max_val, max_val)
            
            # 最大メンバーシップの言語値に分類
            memberships = {"low": low_membership, "medium": medium_membership, "high": high_membership}
            best_linguistic_value = max(memberships, key=memberships.get)
            
            splits[best_linguistic_value][0].append(sample)
            splits[best_linguistic_value][1].append(y[i])
        
        # 情報利得を計算
        total_samples = len(X)
        weighted_impurity = 0.0
        
        for linguistic_value, (X_subset, y_subset) in splits.items():
            if len(y_subset) > 0:
                subset_weight = len(y_subset) / total_samples
                subset_impurity = self._calculate_impurity(y_subset)
                weighted_impurity += subset_weight * subset_impurity
        
        current_impurity = self._calculate_impurity(y)
        information_gain = current_impurity - weighted_impurity
        
        return information_gain, splits
    
    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """三角形メンバーシップ関数"""
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if b != a else 0.0
        else:
            return (c - x) / (c - b) if c != b else 0.0
    
    def _calculate_impurity(self, y: List[str]) -> float:
        """不純度計算（ジニ不純度）"""
        if not y:
            return 0.0
            
        class_counts = Counter(y)
        total = len(y)
        impurity = 1.0
        
        for count in class_counts.values():
            prob = count / total
            impurity -= prob ** 2
        
        return impurity
    
    def predict(self, instance: Dict[str, float]) -> Dict[str, Any]:
        """単一インスタンスの予測"""
        if self.root is None:
            raise ValueError("決定木が学習されていません")
        
        start_time = time.time()
        prediction = self.root.predict(instance)
        prediction_time = time.time() - start_time
        
        prediction["prediction_time"] = prediction_time
        return prediction
    
    def predict_batch(self, X: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """バッチ予測"""
        return [self.predict(instance) for instance in X]
    
    def _calculate_accuracy(self, X: List[Dict[str, float]], y: List[str]) -> float:
        """予測精度を計算"""
        if not X or self.root is None:
            return 0.0
        
        correct = 0
        for i, instance in enumerate(X):
            prediction = self.predict(instance)
            if prediction["predicted_class"] == y[i]:
                correct += 1
        
        return correct / len(X)
    
    def _extract_rules(self):
        """決定木からファジィルールを抽出"""
        if self.root is None:
            return
        
        self.rules = []
        self._extract_rules_recursive(self.root, [])
        
        logger.info(f"ファジィルール {len(self.rules)} 個を抽出しました")
    
    def _extract_rules_recursive(self, node: FuzzyTreeNode, conditions: List[FuzzyCondition]):
        """再帰的にルール抽出"""
        if isinstance(node, FuzzyLeafNode):
            # 葉ノードに到達：ルールを生成
            rule = FuzzyRule(
                conditions=conditions.copy(),
                conclusion=node.predicted_class,
                confidence=node.confidence,
                support=node.samples_count / self.training_samples,
                rule_id=len(self.rules)
            )
            self.rules.append(rule)
        
        elif isinstance(node, FuzzyInternalNode):
            # 内部ノード：各子ノードに対して条件を追加して再帰
            for linguistic_value, child_node in node.children.items():
                condition = FuzzyCondition(
                    attribute=node.attribute,
                    linguistic_value=linguistic_value,
                    membership_degree=1.0,  # 簡略化
                    threshold=0.5
                )
                new_conditions = conditions + [condition]
                self._extract_rules_recursive(child_node, new_conditions)
    
    def get_rules_as_text(self) -> List[str]:
        """ルールをテキスト形式で取得"""
        text_rules = []
        
        for rule in self.rules:
            conditions_text = " AND ".join([
                f"{cond.attribute} is {cond.linguistic_value}"
                for cond in rule.conditions
            ])
            
            rule_text = f"IF {conditions_text} THEN compatibility = {rule.conclusion} (confidence: {rule.confidence:.3f})"
            text_rules.append(rule_text)
        
        return text_rules
    
    def get_tree_summary(self) -> Dict[str, Any]:
        """決定木のサマリー情報を取得"""
        return {
            "tree_id": self.tree_id,
            "node_count": self.node_count,
            "max_depth": self.max_depth,
            "training_samples": self.training_samples,
            "training_accuracy": self.training_accuracy,
            "feature_count": len(self.feature_names),
            "class_count": len(self.class_names),
            "rules_count": len(self.rules),
            "features": self.feature_names
        }